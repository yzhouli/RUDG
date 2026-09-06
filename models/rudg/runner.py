"""Leakage-safe training and evaluation runner for RUDG.

The runner implements fixed-view validation, checkpoint selection, and a
post-freeze test protocol. The model itself is registered by
``models.rudg.model`` at runtime.

Key evaluation safeguard: validation blocks are sampled once and then frozen
for the complete run.  Neighbour sampling in DGL 1.1 is not reproducible by a
simple seed reset, so this prevents stochastic validation views from choosing
the checkpoint or threshold.  Validation-only mode never reads test nodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DGL_ROOT = ROOT / ".tmp" / "dgl-cu118"
if LOCAL_DGL_ROOT.exists():
    sys.path.insert(0, str(LOCAL_DGL_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / ".tmp" / "dgl"))
_requested_cpu_threads = int(os.environ.get("RUDG_CPU_THREADS", "0"))
if _requested_cpu_threads > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(_requested_cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(_requested_cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_requested_cpu_threads))

if os.name == "nt":
    _cuda_runtime = os.environ.get(
        "DGL_CUDA_RUNTIME_DIR", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    )
    if os.path.isdir(_cuda_runtime):
        os.add_dll_directory(_cuda_runtime)

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn.functional import edge_softmax
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(ROOT / "src"))

from fraudlab.evaluation import choose_threshold, evaluate, save_prediction_artifact  # noqa: E402
from fraudlab.splits import load_split_manifest  # noqa: E402


DATASET_CONFIG = {
    "amazon": {"fanouts": (28, 12), "batch_size": 1024, "eval_batch_size": 4096},
    "yelpchi": {"fanouts": (20, 12), "batch_size": 1024, "eval_batch_size": 4096},
    "dgraphfin": {"fanouts": (15, 10), "batch_size": 2048, "eval_batch_size": 4096},
    "tfinance": {"fanouts": (16, 10), "batch_size": 1024, "eval_batch_size": 4096},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_training_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_graph(dataset: str, seed: int) -> tuple[dgl.DGLGraph, dict, Path]:
    split_path = ROOT / "data/splits" / f"{dataset}_stratified_80_10_10_seed{seed}.npz"
    split_meta = json.loads(split_path.with_suffix(".json").read_text(encoding="utf-8"))
    cache_dir = "gaap_unified" if dataset == "dgraphfin" else "gpa_unified"
    graph_path = (
        ROOT / "data/processed" / cache_dir
        / f"{dataset}_seed{seed}_{split_meta['split_sha256'][:12]}.dgl"
    )
    if not graph_path.exists():
        raise FileNotFoundError(f"shared graph cache not found: {graph_path}")
    graph = dgl.load_graphs(str(graph_path))[0][0]
    # Preserve original source-to-destination edges; self loops only retain
    # the centre-node signal when a sampled neighbourhood is unreliable.
    return dgl.add_self_loop(graph), split_meta, graph_path


def attach_structural_features(graph: dgl.DGLGraph) -> None:
    """Add label-free, direction-aware degree descriptors to the CPU graph."""
    in_degree = torch.log1p(graph.in_degrees().float())
    out_degree = torch.log1p(graph.out_degrees().float())
    balance = (in_degree - out_degree) / (in_degree + out_degree + 1.0)
    structural = torch.stack([in_degree, out_degree, balance], dim=-1)
    structural = (structural - structural.mean(dim=0)) / structural.std(dim=0).clamp_min(1e-6)
    graph.ndata["rudg_structural"] = structural


def cap_indices(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.RandomState(seed)
    selected = []
    for cls in (0, 1):
        members = indices[labels[indices] == cls]
        count = max(1, round(limit * len(members) / len(indices)))
        selected.append(rng.choice(members, min(count, len(members)), replace=False))
    result = np.concatenate(selected)
    return np.sort(rng.choice(result, min(limit, len(result)), replace=False).astype(np.int64))


def weighted_focal_bce(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor, gamma: float
) -> torch.Tensor:
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if gamma <= 0.0:
        return base.mean()
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    return ((1.0 - p_t).pow(gamma) * base).mean()


class UncertaintyDirectedConvV2(nn.Module):
    """Reliability-normalised, directional aggregation with a residual gate."""

    def __init__(self, hidden: int, dropout: float, max_aug_drop: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden, hidden, bias=False)
        self.value_linear = nn.Linear(hidden, hidden, bias=False)
        self.node_uncertainty = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.edge_uncertainty = nn.Sequential(
            nn.Linear(hidden * 3 + 9, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.direction_score = nn.Sequential(
            nn.Linear(hidden * 2 + 9, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.gate = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.raw_uncertainty_penalty = nn.Parameter(torch.tensor(0.5))
        self.max_aug_drop = max_aug_drop

    def forward(
        self, block: dgl.DGLGraph, h: torch.Tensor, structural: torch.Tensor, augment: bool
    ) -> tuple[torch.Tensor, dict]:
        h_dst = h[: block.num_dst_nodes()]
        structural_dst = structural[: block.num_dst_nodes()]
        source, target = block.edges(order="eid")
        source_h, target_h = h[source], h_dst[target]
        source_s, target_s = structural[source], structural_dst[target]
        source_u = torch.sigmoid(self.node_uncertainty(torch.cat([source_h, source_s], dim=-1))).squeeze(-1)
        target_u = torch.sigmoid(self.node_uncertainty(torch.cat([target_h, target_s], dim=-1))).squeeze(-1)
        relation_features = torch.cat(
            [source_h, target_h, (source_h - target_h).abs(), source_s, target_s, (source_s - target_s).abs()], dim=-1
        )
        relation_u = torch.sigmoid(self.edge_uncertainty(relation_features)).squeeze(-1)
        score_inputs = torch.cat(
            [source_h, target_h, source_s, target_s, source_u[:, None], target_u[:, None], relation_u[:, None]],
            dim=-1,
        )
        reliability = (1.0 - relation_u).clamp_min(0.05)
        score = self.direction_score(score_inputs).squeeze(-1)
        score = score + reliability.log() - F.softplus(self.raw_uncertainty_penalty) * relation_u
        attention = edge_softmax(block, score)
        if augment:
            # Independent reliability-aware dropout yields the two contrastive
            # graph views.  Low-confidence relations are perturbed first.
            probability = self.max_aug_drop * relation_u.detach()
            keep = (torch.rand_like(probability) >= probability).to(attention.dtype)
            attention = attention * keep / (1.0 - probability).clamp_min(0.1)
        with block.local_scope():
            block.srcdata["value"] = self.value_linear(h)
            block.edata["attention"] = attention
            block.update_all(fn.u_mul_e("value", "attention", "message"), fn.sum("message", "neighbour"))
            neighbour = block.dstdata["neighbour"]
        self_message = self.self_linear(h_dst)
        message_gate = torch.sigmoid(self.gate(torch.cat([self_message, neighbour], dim=-1)))
        output = self.norm(self_message + message_gate * neighbour)
        output = self.dropout(F.gelu(output))
        disagreement = 0.5 * (1.0 - F.cosine_similarity(source_h, target_h, dim=-1))
        calibration = F.mse_loss(relation_u, disagreement.detach())
        return output, {
            "mean_node_uncertainty": 0.5 * (source_u.mean() + target_u.mean()),
            "mean_relation_uncertainty": relation_u.mean(),
            "calibration_loss": calibration,
        }


class BaseRUDG(nn.Module):
    """Fallback model interface used when this runner is invoked directly."""

    model_name = "RUDG-base"
    validation_variant = "RUDG-base (validation selection)"
    formal_variant = "RUDG-base"
    file_stem = "rudg_base"
    validation_view_kind = "rudg_base_fixed_validation_view"

    def __init__(self, input_dim: int, hidden: int, dropout: float, max_aug_drop: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim + 3, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.layers = nn.ModuleList(
            [
                UncertaintyDirectedConvV2(hidden, dropout, max_aug_drop),
                UncertaintyDirectedConvV2(hidden, dropout, max_aug_drop),
            ]
        )
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def encode_view(self, blocks: list[dgl.DGLGraph], augment: bool) -> tuple[torch.Tensor, dict]:
        structural = blocks[0].srcdata["rudg_structural"].float()
        h = self.input(torch.cat([blocks[0].srcdata["feat"].float(), structural], dim=-1))
        root_count = blocks[-1].num_dst_nodes()
        root = h[:root_count]
        layer_stats = []
        for layer, block in zip(self.layers, blocks):
            h, stats = layer(block, h, structural, augment)
            # The target nodes of one DGL block are precisely the source nodes
            # of the next one, including their matching structural descriptors.
            structural = structural[: block.num_dst_nodes()]
            layer_stats.append(stats)
        return self.fuse(torch.cat([h, root], dim=-1)), {
            "mean_node_uncertainty": torch.stack([item["mean_node_uncertainty"] for item in layer_stats]).mean(),
            "mean_relation_uncertainty": torch.stack([item["mean_relation_uncertainty"] for item in layer_stats]).mean(),
            "calibration_loss": torch.stack([item["calibration_loss"] for item in layer_stats]).mean(),
        }

    def forward(self, blocks: list[dgl.DGLGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        z1, stats1 = self.encode_view(blocks, self.training)
        z2, stats2 = self.encode_view(blocks, self.training)
        logits = self.classifier(0.5 * (z1 + z2)).squeeze(-1)
        stats = {
            key: 0.5 * (stats1[key] + stats2[key])
            for key in ("mean_node_uncertainty", "mean_relation_uncertainty", "calibration_loss")
        }
        return logits, self.projector(z1), self.projector(z2), stats


def symmetric_info_nce(z1: torch.Tensor, z2: torch.Tensor, size: int, tau: float) -> torch.Tensor:
    if size < 2:
        return torch.zeros((), device=z1.device)
    chosen = torch.randperm(len(z1), device=z1.device)[:size]
    first, second = F.normalize(z1[chosen], dim=-1), F.normalize(z2[chosen], dim=-1)
    targets = torch.arange(size, device=z1.device)
    logits = first @ second.T / tau
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def freeze_evaluation_batches(
    graph: dgl.DGLGraph, sampler: dgl.dataloading.BlockSampler, indices: np.ndarray, batch_size: int
) -> list[tuple[np.ndarray, list[dgl.DGLGraph]]]:
    """Sample a reference evaluation view once; reuse it for every epoch."""
    loader = dgl.dataloading.DataLoader(
        graph, indices, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
    )
    frozen = []
    for _, output_nodes, blocks in loader:
        frozen.append((output_nodes.numpy().copy(), blocks))
    order = np.concatenate([output_nodes for output_nodes, _ in frozen])
    if len(order) != len(indices) or not np.array_equal(order, indices):
        raise RuntimeError("evaluation loader unexpectedly changed node order")
    return frozen


def load_or_freeze_validation_batches(
    cache_path: Path | None,
    graph: dgl.DGLGraph,
    sampler: dgl.dataloading.BlockSampler,
    indices: np.ndarray,
    batch_size: int,
    metadata: dict,
) -> tuple[list[tuple[np.ndarray, list[dgl.DGLGraph]]], str]:
    """Reuse an on-disk validation view so candidates see identical blocks."""
    if cache_path is None:
        return freeze_evaluation_batches(graph, sampler, indices, batch_size), "in_memory_frozen_once"
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    expected = dict(metadata)
    expected["batch_size"] = batch_size
    if cache_path.exists() or metadata_path.exists():
        if not (cache_path.exists() and metadata_path.exists()):
            raise RuntimeError("validation cache graph and metadata must either both exist or both be absent")
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected:
            raise RuntimeError("validation cache metadata does not match this experiment")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "rudg_block_cache_v1":
            raise RuntimeError("validation cache format is not supported")
        frozen = []
        for batch in payload["batches"]:
            blocks = []
            for block_payload in batch["blocks"]:
                block = dgl.create_block(
                    (block_payload["source"], block_payload["target"]),
                    num_src_nodes=int(block_payload["num_src_nodes"]),
                    num_dst_nodes=int(block_payload["num_dst_nodes"]),
                )
                for name, value in block_payload["srcdata"].items():
                    block.srcdata[name] = value
                blocks.append(block)
            frozen.append((batch["output_nodes"].numpy().astype(np.int64), blocks))
        loaded_order = np.concatenate([nodes for nodes, _ in frozen])
        if not np.array_equal(loaded_order, indices):
            raise RuntimeError("validation cache node order does not match the requested split")
        return frozen, f"disk_cache:{cache_path.relative_to(ROOT)}"

    frozen = freeze_evaluation_batches(graph, sampler, indices, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    persisted_batches = []
    for output_nodes, blocks in frozen:
        persisted_blocks = []
        for index, block in enumerate(blocks):
            source, target = block.edges(order="eid")
            srcdata = {}
            if index == 0:
                # v2 reads only the first block's input features/structural
                # roles.  Omitting copied labels and unused data keeps the
                # validation cache compact and cannot create label leakage.
                srcdata = {
                    "feat": block.srcdata["feat"].cpu(),
                    "rudg_structural": block.srcdata["rudg_structural"].cpu(),
                }
            persisted_blocks.append(
                {
                    "source": source.cpu(), "target": target.cpu(),
                    "num_src_nodes": block.num_src_nodes(), "num_dst_nodes": block.num_dst_nodes(),
                    "srcdata": srcdata,
                }
            )
        persisted_batches.append({"output_nodes": torch.as_tensor(output_nodes, dtype=torch.int64), "blocks": persisted_blocks})
    torch.save({"format": "rudg_block_cache_v1", "batches": persisted_batches}, cache_path)
    metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return frozen, f"created_disk_cache:{cache_path.relative_to(ROOT)}"


@torch.no_grad()
def score_frozen_nodes(
    model: BaseRUDG, batches: list[tuple[np.ndarray, list[dgl.DGLGraph]]], device: torch.device
) -> np.ndarray:
    model.eval()
    chunks = []
    for _, blocks in batches:
        device_blocks = [block.to(device) for block in blocks]
        logits, _, _, _ = model(device_blocks)
        chunks.append(torch.sigmoid(logits).cpu())
    return torch.cat(chunks).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="amazon")
    parser.add_argument("--mode", choices=("validation", "formal"), default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Dataset split seed; defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--fanouts", default="")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-aug-drop", type=float, default=0.15)
    parser.add_argument("--contrast-weight", type=float, default=0.03)
    parser.add_argument("--calibration-weight", type=float, default=0.03)
    parser.add_argument("--calibration-representation-mix", type=float, default=0.75)
    parser.add_argument("--reliability-floor", type=float, default=0.05)
    parser.add_argument("--uncertainty-penalty-init", type=float, default=0.5)
    parser.add_argument("--contrast-batch", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--feature-bins", type=int, default=8)
    parser.add_argument("--memory-tokens", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--gaap-residual-scale", type=float, default=0.30)
    parser.add_argument("--risk-residual-scale", type=float, default=0.30)
    parser.add_argument(
        "--ablation",
        default="none",
        choices=(
            "none",
            "without_calibration",
            "uniform_perturbation",
            "without_contrast",
            "symmetric_propagation",
            "without_reliability",
            "without_structure",
            "without_soft_binning",
            "without_prototypes",
        ),
    )
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--post-freeze-state", type=Path)
    args = parser.parse_args()

    # The public model entry point registers its model class before calling
    # this leakage-safe training and frozen-evaluation harness.
    model_class = BaseRUDG
    model_name = getattr(model_class, "model_name", "RUDG-base")
    validation_variant = getattr(model_class, "validation_variant", f"{model_name} (validation selection)")
    formal_variant = getattr(model_class, "formal_variant", f"{model_name} (Proposed)")
    file_stem = getattr(model_class, "file_stem", model_name.lower().replace("-", "_"))
    validation_view_kind = getattr(
        model_class, "validation_view_kind", f"{file_stem}_fixed_validation_view"
    )

    if args.contrast_weight < 0.0 or args.calibration_weight < 0.0:
        raise ValueError("contrast and uncertainty-calibration weights must be non-negative")
    if args.ablation != "without_contrast" and args.contrast_weight == 0.0:
        raise ValueError(f"{model_name} requires positive contrast weight outside its contrast ablation")
    if args.ablation != "without_calibration" and args.calibration_weight == 0.0:
        raise ValueError(f"{model_name} requires positive calibration weight outside its calibration ablation")
    if args.ablation == "without_contrast" and args.contrast_weight != 0.0:
        raise ValueError("without_contrast requires --contrast-weight 0")
    if args.ablation == "without_calibration" and args.calibration_weight != 0.0:
        raise ValueError("without_calibration requires --calibration-weight 0")
    if not 0.0 <= args.max_aug_drop < 1.0:
        raise ValueError("max-aug-drop must be in [0, 1)")
    if args.focal_gamma < 0.0:
        raise ValueError("focal-gamma must be non-negative")
    if args.pos_weight_scale <= 0.0:
        raise ValueError("pos-weight-scale must be positive")
    if args.layers < 1:
        raise ValueError("layers must be at least one")
    if not 0.0 <= args.calibration_representation_mix <= 1.0:
        raise ValueError("calibration-representation-mix must be in [0, 1]")
    if not 0.0 < args.reliability_floor < 1.0:
        raise ValueError("reliability-floor must be in (0, 1)")

    if _requested_cpu_threads > 0:
        torch.set_num_threads(_requested_cpu_threads)
    started = time.time()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested unavailable device: {device}")

    config = DATASET_CONFIG[args.dataset]
    requested_layers = args.layers if getattr(model_class, "supports_variable_layers", False) else 2
    fanouts = tuple(int(value) for value in args.fanouts.split(",") if value) or config["fanouts"]
    if len(fanouts) != requested_layers or any(value <= 0 for value in fanouts):
        raise ValueError(f"fanouts must contain {requested_layers} positive comma-separated integers")
    batch_size = args.batch_size or config["batch_size"]
    eval_batch_size = args.eval_batch_size or config["eval_batch_size"]
    split_seed = args.seed if args.split_seed is None else args.split_seed
    graph, split_meta, graph_path = load_graph(args.dataset, split_seed)
    attach_structural_features(graph)
    labels_np = graph.ndata["label"].long().reshape(-1).numpy()
    split_path = ROOT / "data/splits" / f"{args.dataset}_stratified_80_10_10_seed{split_seed}.npz"
    split = load_split_manifest(split_path, dataset=args.dataset, seed=split_seed)
    train_np, valid_np, test_np = split["train_idx"], split["valid_idx"], split["test_idx"]
    if args.smoke_limit:
        train_np = cap_indices(train_np, labels_np, args.smoke_limit, args.seed)
        valid_np = cap_indices(valid_np, labels_np, max(64, args.smoke_limit // 2), args.seed + 1)
        if args.mode == "formal":
            test_np = cap_indices(test_np, labels_np, max(64, args.smoke_limit // 2), args.seed + 2)

    if hasattr(model_class, "from_args"):
        model = model_class.from_args(
            graph.ndata["feat"].shape[1], args.hidden, args.dropout, args.max_aug_drop, args
        ).to(device)
    else:
        model = model_class(graph.ndata["feat"].shape[1], args.hidden, args.dropout, args.max_aug_drop).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_labels = labels_np[train_np]
    positives = int(train_labels.sum())
    pos_weight = torch.tensor(
        [args.pos_weight_scale * (len(train_labels) - positives) / max(1, positives)],
        dtype=torch.float32,
        device=device,
    )
    sampler = dgl.dataloading.MultiLayerNeighborSampler(list(fanouts))
    # This is the only validation graph view.  No test blocks are sampled in
    # validation mode, including while early stopping.
    frozen_valid, validation_view = load_or_freeze_validation_batches(
        args.validation_cache,
        graph,
        sampler,
        valid_np,
        eval_batch_size,
        {
            "kind": validation_view_kind, "dataset": args.dataset, "seed": split_seed,
            "split_sha256": split_meta["split_sha256"], "fanouts": list(fanouts),
            "smoke_limit": args.smoke_limit,
        },
    )
    state_path = args.state_output or ROOT / "outputs/checkpoints" / f"{file_stem}_{args.dataset}_seed{args.seed}.state.pt"
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    signature = {
        "model": model_name, "dataset": args.dataset, "mode": args.mode, "seed": args.seed,
        "split_sha256": split_meta["split_sha256"], "hidden": args.hidden, "fanouts": fanouts,
        "batch_size": batch_size, "eval_batch_size": eval_batch_size, "dropout": args.dropout,
        "max_aug_drop": args.max_aug_drop, "contrast_weight": args.contrast_weight,
        "calibration_weight": args.calibration_weight, "contrast_batch": args.contrast_batch, "tau": args.tau,
        "focal_gamma": args.focal_gamma, "lr": args.lr, "weight_decay": args.weight_decay,
        "smoke_limit": args.smoke_limit,
    }
    if args.split_seed is not None:
        signature["split_seed"] = split_seed
    if args.pos_weight_scale != 1.0:
        # Preserve compatibility with pre-existing default-weight checkpoints.
        signature["pos_weight_scale"] = args.pos_weight_scale
    if hasattr(model, "experiment_signature"):
        signature["architecture"] = model.experiment_signature()
    best_ap, best_epoch, best_state, wait, early_stopped = -np.inf, 0, None, 0, False
    history, start_epoch = [], 1
    if args.resume_state and args.post_freeze_state:
        raise ValueError("resume-state and post-freeze-state are mutually exclusive")
    checkpoint_origin = None
    checkpoint_input = args.post_freeze_state or args.resume_state
    if checkpoint_input:
        checkpoint_path = checkpoint_input if checkpoint_input.is_absolute() else ROOT / checkpoint_input
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_signature = dict(saved.get("signature", {}))
        if args.post_freeze_state:
            if args.mode != "formal":
                raise ValueError("post-freeze-state is allowed only in formal mode")
            # Configuration selection was completed in validation mode.  The
            # state contains its best validation checkpoint; changing only the
            # execution mode does not change architecture or hyperparameters.
            saved_signature["mode"] = "formal"
        signature_compatible = saved_signature == signature
        compatibility_check = getattr(model_class, "checkpoint_signature_compatible", None)
        if not signature_compatible and compatibility_check is not None:
            signature_compatible = bool(compatibility_check(saved_signature, signature))
        if not signature_compatible:
            raise RuntimeError(f"{model_name} checkpoint signature does not match this run")
        best_ap, best_epoch, best_state = float(saved["best_ap"]), int(saved["best_epoch"]), saved["best_state"]
        wait, early_stopped, history = int(saved["wait"]), bool(saved["early_stopped"]), list(saved["history"])
        if args.post_freeze_state:
            if not early_stopped:
                raise RuntimeError("post-freeze-state must originate from an early-stopped validation selection run")
            model.load_state_dict(best_state)
            start_epoch = args.max_epochs + 1
            checkpoint_origin = str(checkpoint_path.relative_to(ROOT))
            print(json.dumps({"post_freeze_checkpoint": checkpoint_origin, "best_epoch": best_epoch, "best_ap": best_ap}), flush=True)
        else:
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            start_epoch = int(saved["epoch"]) + 1
            print(json.dumps({"resumed_from_epoch": start_epoch - 1, "best_epoch": best_epoch, "best_ap": best_ap}), flush=True)
            # A resumed job may use a stricter patience limit than the
            # interrupted process.  Honor the already accumulated wait count
            # before scheduling another epoch so the selected checkpoint and
            # the reported training length remain protocol-consistent.
            if wait >= args.patience:
                early_stopped = True
                save_training_state(state_path, {
                    "signature": signature, "epoch": start_epoch - 1,
                    "model_state": cpu_state_dict(model),
                    "optimizer_state": cpu_tree(optimizer.state_dict()),
                    "best_ap": best_ap, "best_epoch": best_epoch,
                    "best_state": best_state, "wait": wait,
                    "early_stopped": early_stopped, "history": history,
                })
                print(json.dumps({"early_stopped_on_resume": True, "wait": wait, "patience": args.patience}), flush=True)

    for epoch in range(start_epoch, args.max_epochs + 1):
        if early_stopped:
            break
        model.train()
        train_loader = dgl.dataloading.DataLoader(
            graph, train_np, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
        )
        total_loss = total_cls = total_contrast = total_calibration = 0.0
        examples = 0
        for _, output_nodes, blocks in train_loader:
            device_blocks = [block.to(device) for block in blocks]
            targets = torch.as_tensor(labels_np[output_nodes.numpy()], dtype=torch.float32, device=device)
            logits, view1, view2, stats = model(device_blocks)
            classification_loss = weighted_focal_bce(logits, targets, pos_weight, args.focal_gamma)
            contrast_loss = symmetric_info_nce(view1, view2, min(args.contrast_batch, len(targets)), args.tau)
            loss = classification_loss + args.contrast_weight * contrast_loss + args.calibration_weight * stats["calibration_loss"]
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                # A sampled block must never be allowed to contaminate the
                # optimizer state.  The event is still deterministic for a
                # fixed seed and uses no validation/test labels.
                continue
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(targets)
            examples += count
            total_loss += float(loss.detach().cpu()) * count
            total_cls += float(classification_loss.detach().cpu()) * count
            total_contrast += float(contrast_loss.detach().cpu()) * count
            total_calibration += float(stats["calibration_loss"].detach().cpu()) * count

        valid_scores = score_frozen_nodes(model, frozen_valid, device)
        valid_ap = float(average_precision_score(labels_np[valid_np], valid_scores))
        improved = valid_ap > best_ap + 1e-6
        if improved:
            best_ap, best_epoch, best_state, wait = valid_ap, epoch, cpu_state_dict(model), 0
        else:
            wait += 1
        early_stopped = wait >= args.patience
        event = {
            "epoch": epoch, "train_loss": total_loss / max(1, examples),
            "classification_loss": total_cls / max(1, examples), "contrast_loss": total_contrast / max(1, examples),
            "uncertainty_calibration_loss": total_calibration / max(1, examples),
            "valid_pr_auc": valid_ap, "improved": improved,
        }
        history.append(event)
        save_training_state(state_path, {
            "signature": signature, "epoch": epoch, "model_state": cpu_state_dict(model),
            "optimizer_state": cpu_tree(optimizer.state_dict()), "best_ap": best_ap, "best_epoch": best_epoch,
            "best_state": best_state, "wait": wait, "early_stopped": early_stopped, "history": history,
        })
        print(json.dumps(event), flush=True)
        if early_stopped:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    valid_scores = score_frozen_nodes(model, frozen_valid, device)
    model_profile = {
        "hidden": args.hidden,
        "layers": requested_layers,
        "fanouts": list(fanouts),
        "dropout": args.dropout,
    }
    if hasattr(model, "report_profile"):
        model_profile.update(model.report_profile())
    validation_result = {
        "model": model_name, "variant": validation_variant, "dataset": args.dataset,
        "device": str(device), "seed": args.seed, "split_seed": split_seed,
        "split_sha256": split_meta["split_sha256"],
        "maximum_epochs": args.max_epochs, "patience": args.patience, "epochs_ran": len(history),
        "best_epoch": best_epoch, "early_stopped": early_stopped, "validation_best_pr_auc": best_ap,
        "model_profile": model_profile,
        "loss_weights": {"fraud": 1.0, "contrast": args.contrast_weight, "uncertainty_calibration": args.calibration_weight},
        "search_parameters": {
            "tau": args.tau,
            "max_aug_drop": args.max_aug_drop,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "feature_bins": args.feature_bins,
            "memory_tokens": args.memory_tokens,
            "attention_heads": args.attention_heads,
            "dropout": args.dropout,
            "calibration_representation_mix": args.calibration_representation_mix,
            "reliability_floor": args.reliability_floor,
            "uncertainty_penalty_init": args.uncertainty_penalty_init,
            "pos_weight_scale": args.pos_weight_scale,
        },
        "focal_gamma": args.focal_gamma,
        "classification_loss": {
            "name": "weighted_focal_bce",
            "focal_gamma": args.focal_gamma,
            "pos_weight_scale": args.pos_weight_scale,
        },
        "evaluation_protocol": "one frozen sampled validation view, reused at every epoch",
        "validation_view": validation_view,
        "label_usage_audit": "labels are used only for train loss and validation early stopping/threshold; test is untouched in validation mode",
        "directed_graph_policy": "source-to-destination messages only; self-loops are added without reverse edges",
        "memory_policy": {
            "graph_residency": "cpu",
            "gpu_transfer": "sampled_two_hop_dgl_blocks_only",
            "cpu_thread_limit": _requested_cpu_threads or None,
            "cpu_thread_policy": (
                "explicit RUDG_CPU_THREADS limit"
                if _requested_cpu_threads > 0
                else "runtime default; no RUDG CPU-core limit"
            ),
        },
        "training_state": str(state_path.relative_to(ROOT)), "runtime_seconds": time.time() - started,
        "history": history, "formal_performance": False, "post_freeze_checkpoint": checkpoint_origin,
    }
    if hasattr(model, "design_lineage"):
        validation_result["design_lineage"] = model.design_lineage
    if args.mode == "validation":
        if args.output is None:
            args.output = ROOT / "outputs/validation" / f"{file_stem}_{args.dataset}_validation.json"
        if not args.output.is_absolute():
            args.output = ROOT / args.output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(validation_result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(validation_result, ensure_ascii=False), flush=True)
        return

    # Formal mode is reached only after a configuration was selected without
    # reading the test split.  The test graph view is sampled once post-freeze.
    frozen_test = freeze_evaluation_batches(graph, sampler, test_np, eval_batch_size)
    test_scores = score_frozen_nodes(model, frozen_test, device)
    threshold = choose_threshold(labels_np[valid_np], valid_scores)
    metrics = evaluate(labels_np[test_np], test_scores, threshold)
    formal = bool(args.smoke_limit == 0 and args.max_epochs >= 100 and args.patience >= 10 and early_stopped)
    if args.output is None:
        destination = ROOT / "outputs" / (args.dataset if formal else "checks")
        args.output = destination / (
            f"{file_stem}_proposed_unified.json" if formal else f"{file_stem}_{args.dataset}_formal_smoke.json"
        )
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_name(f"{args.output.stem}_predictions.npz")
    save_prediction_artifact(prediction_path, labels_np[test_np], test_scores, threshold, test_np)
    result = dict(validation_result)
    result.update({
        "variant": formal_variant, "training_sufficient": formal, "formal_performance": formal,
        "sample_counts": {"train": len(train_np), "valid": len(valid_np), "test": len(test_np)},
        "graph_cache": str(graph_path.relative_to(ROOT)), "threshold": threshold, "test": metrics,
        "evaluation_protocol": "frozen validation view for checkpoint/threshold; one post-freeze frozen test view",
        "label_usage_audit": "labels are used only for train loss, validation early stopping/threshold, and frozen-checkpoint test evaluation",
        "prediction_artifact": str(prediction_path.relative_to(ROOT)),
    })
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
