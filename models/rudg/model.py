"""RUDG: robust uncertainty-aware directed graph learning.

The model keeps RUDG's two mandatory scientific constraints: relation
uncertainty controls graph augmentation, and two independently augmented views
are aligned with contrastive learning.  GAAP contributes feature-wise soft
binning and global reference memory; RGTAN contributes multi-head directional
attention and structural-risk gating.  No label embedding is present: every
message, risk descriptor and memory key is computed from attributes or graph
structure only.

The leakage-safe training, fixed validation view, post-freeze test evaluation,
and CPU graph residency are provided by ``models.rudg.runner``. CPU threads follow
the runtime default unless ``RUDG_CPU_THREADS`` is explicitly supplied.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rudg import runner  # noqa: E402


dgl = runner.dgl
fn = runner.fn
torch = runner.torch
nn = runner.nn
F = runner.F
edge_softmax = runner.edge_softmax


class FeaturewiseSoftBinning(nn.Module):
    """GAAP-inspired differentiable per-feature piecewise encoding.

    Each numeric field obtains its own bin embeddings.  A raw linear branch is
    retained so saturated or nearly constant fields never lose their original
    signal.  The operation is batch-local and does not fit statistics on the
    validation or test split.
    """

    def __init__(self, input_dim: int, hidden: int, bins: int, dropout: float) -> None:
        super().__init__()
        if bins < 3:
            raise ValueError("feature-bins must be at least 3")
        self.input_dim = input_dim
        self.bins = bins
        self.center = nn.Parameter(torch.zeros(input_dim))
        self.raw_scale = nn.Parameter(torch.zeros(input_dim))
        self.bin_embeddings = nn.Parameter(torch.empty(input_dim, bins, hidden))
        nn.init.normal_(self.bin_embeddings, std=1.0 / math.sqrt(hidden))
        self.register_buffer("knots", torch.linspace(0.0, 1.0, bins))
        self.raw_projection = nn.Linear(input_dim + 3, hidden)
        self.bin_norm = nn.LayerNorm(hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, structural: torch.Tensor, *, use_soft_binning: bool = True
    ) -> torch.Tensor:
        features = torch.nan_to_num(features.float())
        raw = self.raw_projection(torch.cat([features, structural], dim=-1))
        if not use_soft_binning:
            return self.dropout(F.gelu(self.output_norm(raw)))
        scale = F.softplus(self.raw_scale) + 0.25
        unit = torch.sigmoid((features - self.center) * scale)
        width = 1.0 / (self.bins - 1)
        basis = F.relu(1.0 - (unit.unsqueeze(-1) - self.knots).abs() / width)
        basis = basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        binned = torch.einsum("ndb,dbh->nh", basis, self.bin_embeddings)
        binned = self.bin_norm(binned / math.sqrt(max(1, self.input_dim)))
        gate = torch.sigmoid(self.gate(torch.cat([raw, binned], dim=-1)))
        return self.dropout(F.gelu(self.output_norm(raw + gate * binned)))


class StructuralRiskEncoder(nn.Module):
    """RGTAN-style attention over label-free direction/degree descriptors."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        token_dim = max(8, min(24, hidden // 4))
        self.weight = nn.Parameter(torch.empty(3, token_dim))
        self.bias = nn.Parameter(torch.zeros(3, token_dim))
        self.query = nn.Parameter(torch.empty(token_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.query, std=1.0 / math.sqrt(token_dim))
        self.output = nn.Sequential(nn.Linear(token_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, structural: torch.Tensor) -> torch.Tensor:
        tokens = structural.unsqueeze(-1) * self.weight + self.bias
        weights = torch.softmax(torch.einsum("nkr,r->nk", torch.tanh(tokens), self.query), dim=1)
        return self.output(torch.einsum("nk,nkr->nr", weights, tokens))


class MultiHeadUncertaintyDirectedConv(nn.Module):
    """RGTAN-like gated transformer aggregation controlled by RUDG uncertainty."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        dropout: float,
        max_aug_drop: float,
        calibration_representation_mix: float = 0.75,
        reliability_floor: float = 0.05,
        uncertainty_penalty_init: float = 0.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if heads <= 0:
            raise ValueError("attention-heads must be positive")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = math.ceil(hidden / heads)
        self.inner_dim = heads * self.head_dim
        self.max_aug_drop = max_aug_drop
        self.calibration_representation_mix = calibration_representation_mix
        self.reliability_floor = reliability_floor
        self.ablation = ablation
        self.self_linear = nn.Linear(hidden, hidden, bias=False)
        self.query_linear = nn.Linear(hidden, self.inner_dim, bias=False)
        self.key_linear = nn.Linear(hidden, self.inner_dim, bias=False)
        self.value_linear = nn.Linear(hidden, self.inner_dim, bias=False)
        self.output_linear = nn.Identity() if self.inner_dim == hidden else nn.Linear(self.inner_dim, hidden, bias=False)
        self.node_uncertainty = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.edge_uncertainty = nn.Sequential(
            nn.Linear(hidden * 3 + 9, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.direction_bias = nn.Sequential(
            nn.Linear(hidden * 2 + 9, hidden), nn.GELU(), nn.Linear(hidden, heads)
        )
        self.risk_projection = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        self.skip_gate = nn.Linear(hidden * 4, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.raw_uncertainty_penalty = nn.Parameter(torch.tensor(uncertainty_penalty_init))
        if ablation == "symmetric_propagation":
            self.symmetric_projection = nn.Linear(hidden, self.inner_dim, bias=False)

    def forward(
        self,
        block: dgl.DGLGraph,
        h: torch.Tensor,
        structural: torch.Tensor,
        augment: bool,
        *,
        perturbation_policy: str | None = None,
        perturbation_budget: float = 0.0,
        perturbation_seed: int = 0,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        h_dst = h[: block.num_dst_nodes()]
        structural_dst = structural[: block.num_dst_nodes()]
        source, target = block.edges(order="eid")
        source_h, target_h = h[source], h_dst[target]
        source_s, target_s = structural[source], structural_dst[target]

        source_u = torch.sigmoid(
            self.node_uncertainty(torch.cat([source_h, source_s], dim=-1))
        ).squeeze(-1)
        target_u = torch.sigmoid(
            self.node_uncertainty(torch.cat([target_h, target_s], dim=-1))
        ).squeeze(-1)
        if self.ablation == "symmetric_propagation":
            relation_features = torch.cat(
                [
                    0.5 * (source_h + target_h),
                    (source_h - target_h).abs(),
                    source_h * target_h,
                    0.5 * (source_s + target_s),
                    (source_s - target_s).abs(),
                    source_s * target_s,
                ],
                dim=-1,
            )
        else:
            relation_features = torch.cat(
                [source_h, target_h, (source_h - target_h).abs(), source_s, target_s, (source_s - target_s).abs()],
                dim=-1,
            )
        relation_u = torch.sigmoid(self.edge_uncertainty(relation_features)).squeeze(-1)

        if self.ablation == "symmetric_propagation":
            query = self.symmetric_projection(target_h).view(-1, self.heads, self.head_dim)
            key = self.symmetric_projection(source_h).view(-1, self.heads, self.head_dim)
        else:
            query = self.query_linear(target_h).view(-1, self.heads, self.head_dim)
            key = self.key_linear(source_h).view(-1, self.heads, self.head_dim)
        dot_score = (query * key).sum(dim=-1) / math.sqrt(self.head_dim)
        if self.ablation == "symmetric_propagation":
            direction_inputs = torch.cat(
                [
                    0.5 * (source_h + target_h),
                    (source_h - target_h).abs(),
                    0.5 * (source_s + target_s),
                    (source_s - target_s).abs(),
                    (0.5 * (source_u + target_u))[:, None],
                    (source_u - target_u).abs()[:, None],
                    relation_u[:, None],
                ],
                dim=-1,
            )
        else:
            direction_inputs = torch.cat(
                [source_h, target_h, source_s, target_s, source_u[:, None], target_u[:, None], relation_u[:, None]],
                dim=-1,
            )
        reliability = (1.0 - relation_u).clamp_min(self.reliability_floor)
        score = dot_score + self.direction_bias(direction_inputs)
        if self.ablation != "without_reliability":
            score = score + reliability.log().unsqueeze(-1)
        # Keep the learned uncertainty penalty and attention logits in the
        # numerically stable range.  This is especially important for the
        # large-width points in the OAT sweep; it does not alter topology or
        # introduce any label-side signal.
        penalty = F.softplus(self.raw_uncertainty_penalty.clamp(-8.0, 8.0))
        if self.ablation != "without_reliability":
            score = score - penalty * relation_u.unsqueeze(-1)
        score = score.clamp(-20.0, 20.0)
        attention = edge_softmax(block, score)
        if perturbation_policy is not None and perturbation_budget > 0.0:
            if perturbation_policy not in {"uniform", "high_uncertainty", "low_uncertainty"}:
                raise ValueError(f"unknown robustness perturbation: {perturbation_policy}")
            edge_count = int(relation_u.numel())
            drop_count = (
                min(edge_count - 1, max(0, int(round(perturbation_budget * edge_count))))
                if edge_count > 1
                else 0
            )
            if drop_count:
                if perturbation_policy == "high_uncertainty":
                    priority = relation_u.detach()
                elif perturbation_policy == "low_uncertainty":
                    priority = -relation_u.detach()
                else:
                    # A deterministic edge-local hash makes every budget nested
                    # and keeps the two evaluation views exactly identical.
                    edge_id = torch.arange(edge_count, device=relation_u.device, dtype=torch.float32)
                    hash_input = (
                        (source.float() + 1.0) * 12.9898
                        + (target.float() + 1.0) * 78.233
                        + edge_id * 37.719
                        + float(perturbation_seed) * 0.12345
                    )
                    priority = torch.remainder(torch.sin(hash_input) * 43758.5453, 1.0)
                dropped = torch.topk(priority, drop_count, largest=True, sorted=False).indices
                keep = torch.ones(edge_count, device=attention.device, dtype=attention.dtype)
                keep[dropped] = 0.0
                attention = attention * keep.unsqueeze(-1)
                normalizer = torch.zeros(
                    block.num_dst_nodes(), self.heads, device=attention.device, dtype=attention.dtype
                )
                normalizer.index_add_(0, target, attention)
                attention = attention / normalizer[target].clamp_min(1e-8)
        if augment:
            if self.ablation == "uniform_perturbation":
                drop_probability = self.max_aug_drop * relation_u.detach().mean().expand_as(relation_u)
            else:
                drop_probability = self.max_aug_drop * relation_u.detach()
            keep = (torch.rand_like(drop_probability) >= drop_probability).to(attention.dtype)
            attention = attention * keep.unsqueeze(-1) / (1.0 - drop_probability).clamp_min(0.1).unsqueeze(-1)

        with block.local_scope():
            block.srcdata["rudg_value"] = self.value_linear(h).view(-1, self.heads, self.head_dim)
            block.edata["rudg_attention"] = attention.unsqueeze(-1)
            block.update_all(
                fn.u_mul_e("rudg_value", "rudg_attention", "rudg_message"),
                fn.sum("rudg_message", "rudg_neighbour"),
            )
            neighbour = self.output_linear(block.dstdata["rudg_neighbour"].reshape(-1, self.inner_dim))

        self_message = self.self_linear(h_dst)
        risk = self.risk_projection(structural_dst)
        gate_inputs = torch.cat(
            [self_message, neighbour, (self_message - neighbour).abs(), risk], dim=-1
        )
        gate = torch.sigmoid(self.skip_gate(gate_inputs))
        output = self.norm(gate * self_message + (1.0 - gate) * (neighbour + 0.25 * risk))
        output = self.dropout(F.gelu(output))

        representation_disagreement = 0.5 * (
            1.0 - F.cosine_similarity(source_h, target_h, dim=-1)
        )
        structural_disagreement = torch.sigmoid((source_s - target_s).abs().mean(dim=-1))
        calibration_target = (
            self.calibration_representation_mix * representation_disagreement
            + (1.0 - self.calibration_representation_mix) * structural_disagreement
        )
        calibration = F.mse_loss(relation_u, calibration_target.detach())
        statistics = {
            "mean_node_uncertainty": 0.5 * (source_u.mean() + target_u.mean()),
            "mean_relation_uncertainty": relation_u.mean(),
            "calibration_loss": calibration,
        }
        if collect_diagnostics:
            statistics["relation_uncertainty_values"] = relation_u.detach()
            statistics["calibration_target_values"] = calibration_target.detach()
        return output, statistics


class GlobalPrototypeMemory(nn.Module):
    """Memory-efficient GAAP-style global reference attention.

    A small learned prototype bank replaces an all-node GPU history tensor.
    Retrieval entropy acts as memory uncertainty, so diffuse matches receive a
    smaller residual contribution.
    """

    def __init__(self, hidden: int, heads: int, tokens: int) -> None:
        super().__init__()
        if tokens < heads:
            raise ValueError("memory-tokens must be at least attention-heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = math.ceil(hidden / heads)
        self.inner_dim = heads * self.head_dim
        self.tokens = tokens
        self.prototypes = nn.Parameter(torch.empty(tokens, hidden))
        nn.init.normal_(self.prototypes, std=1.0 / math.sqrt(hidden))
        self.query = nn.Linear(hidden, self.inner_dim, bias=False)
        self.key = nn.Linear(hidden, self.inner_dim, bias=False)
        self.value = nn.Linear(hidden, self.inner_dim, bias=False)
        self.output = nn.Identity() if self.inner_dim == hidden else nn.Linear(self.inner_dim, hidden, bias=False)
        self.gate = nn.Linear(hidden * 3, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(local).view(-1, self.heads, self.head_dim)
        key = self.key(self.prototypes).view(self.tokens, self.heads, self.head_dim)
        value = self.value(self.prototypes).view(self.tokens, self.heads, self.head_dim)
        score = torch.einsum("bhd,mhd->bhm", query, key) / math.sqrt(self.head_dim)
        score = score.clamp(-20.0, 20.0)
        attention = torch.softmax(score, dim=-1)
        context = self.output(torch.einsum("bhm,mhd->bhd", attention, value).reshape(-1, self.inner_dim))
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)
        confidence = 1.0 - entropy.mean(dim=-1) / math.log(self.tokens)
        confidence = (0.25 + 0.75 * confidence).clamp(0.25, 1.0)
        gate = torch.sigmoid(self.gate(torch.cat([local, context, (local - context).abs()], dim=-1)))
        output = self.norm(local + confidence.unsqueeze(-1) * gate * context)
        return output, confidence.mean()


class RUDG(nn.Module):
    """RUDG implementation without label-side message channels."""

    model_name = "RUDG"
    validation_variant = "RUDG (validation selection)"
    formal_variant = "RUDG"
    file_stem = "rudg"
    # The cached blocks are model-agnostic (topology + feat + structural only),
    # Validation blocks are frozen once so checkpoint selection does not depend
    # on a newly sampled neighbourhood at each epoch.
    validation_view_kind = "rudg_fixed_validation_view"
    supports_variable_layers = True
    design_lineage = {
        "feature_encoding": ["feature-wise soft binning", "global prototype memory attention"],
        "directed_modeling": ["multi-head directional attention", "label-free structural-risk skip gate"],
        "robust_learning": ["relation uncertainty", "uncertainty-guided two-view augmentation", "symmetric InfoNCE"],
        "excluded": ["label embeddings", "validation/test labels in message passing", "full-graph GPU history"],
    }

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout: float,
        max_aug_drop: float,
        *,
        feature_bins: int = 8,
        memory_tokens: int = 32,
        attention_heads: int = 4,
        layers: int = 2,
        calibration_representation_mix: float = 0.75,
        reliability_floor: float = 0.05,
        uncertainty_penalty_init: float = 0.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if attention_heads <= 0:
            raise ValueError("attention-heads must be positive")
        if layers < 1:
            raise ValueError("layers must be at least one")
        valid_ablations = {
            "none", "without_calibration", "uniform_perturbation", "without_contrast",
            "symmetric_propagation", "without_reliability", "without_structure",
            "without_soft_binning", "without_prototypes",
        }
        if ablation not in valid_ablations:
            raise ValueError(f"unknown RUDG ablation: {ablation}")
        self.feature_bins = feature_bins
        self.memory_tokens = memory_tokens
        self.attention_heads = attention_heads
        self.layer_count = layers
        self.calibration_representation_mix = calibration_representation_mix
        self.reliability_floor = reliability_floor
        self.uncertainty_penalty_init = uncertainty_penalty_init
        self.ablation = ablation
        self.feature_encoder = FeaturewiseSoftBinning(input_dim, hidden, feature_bins, dropout)
        self.risk_encoder = StructuralRiskEncoder(hidden)
        self.input_risk_gate = nn.Linear(hidden * 2, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [
                MultiHeadUncertaintyDirectedConv(
                    hidden,
                    attention_heads,
                    dropout,
                    max_aug_drop,
                    calibration_representation_mix,
                    reliability_floor,
                    uncertainty_penalty_init,
                    ablation,
                )
                for _ in range(layers)
            ]
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.memory = GlobalPrototypeMemory(hidden, attention_heads, memory_tokens)
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    @classmethod
    def from_args(cls, input_dim: int, hidden: int, dropout: float, max_aug_drop: float, args):
        return cls(
            input_dim,
            hidden,
            dropout,
            max_aug_drop,
            feature_bins=args.feature_bins,
            memory_tokens=args.memory_tokens,
            attention_heads=args.attention_heads,
            layers=args.layers,
            calibration_representation_mix=args.calibration_representation_mix,
            reliability_floor=args.reliability_floor,
            uncertainty_penalty_init=args.uncertainty_penalty_init,
            ablation=getattr(args, "ablation", "none"),
        )

    def experiment_signature(self) -> dict:
        signature = {
            "feature_bins": self.feature_bins,
            "memory_tokens": self.memory_tokens,
            "attention_heads": self.attention_heads,
            "layers": self.layer_count,
            "calibration_representation_mix": self.calibration_representation_mix,
            "reliability_floor": self.reliability_floor,
            "uncertainty_penalty_init": self.uncertainty_penalty_init,
            "feature_encoder": "per_feature_soft_piecewise_v1",
            "global_memory": "uncertainty_gated_prototypes_v1",
            "directed_conv": "multihead_structural_risk_v1",
        }
        if self.ablation != "none":
            signature["ablation"] = self.ablation
        return signature

    def report_profile(self) -> dict:
        profile = {
            "feature_bins": self.feature_bins,
            "memory_tokens": self.memory_tokens,
            "attention_heads": self.attention_heads,
            "layers": self.layer_count,
            "calibration_representation_mix": self.calibration_representation_mix,
            "reliability_floor": self.reliability_floor,
            "uncertainty_penalty_init": self.uncertainty_penalty_init,
        }
        if self.ablation != "none":
            profile["ablation"] = self.ablation
        return profile

    @staticmethod
    def checkpoint_signature_compatible(saved: dict, current: dict) -> bool:
        """Accept checkpoints whose fixed two-layer defaults were implicit."""
        saved_normalized = dict(saved)
        saved_architecture = dict(saved_normalized.get("architecture", {}))
        saved_architecture.setdefault("layers", 2)
        saved_architecture.setdefault("calibration_representation_mix", 0.75)
        saved_architecture.setdefault("reliability_floor", 0.05)
        saved_architecture.setdefault("uncertainty_penalty_init", 0.5)
        saved_normalized["architecture"] = saved_architecture
        return saved_normalized == current

    def encode_view(
        self,
        blocks: list[dgl.DGLGraph],
        augment: bool,
        *,
        perturbation_policy: str | None = None,
        perturbation_budget: float = 0.0,
        perturbation_seed: int = 0,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        structural = blocks[0].srcdata["rudg_structural"].float()
        features = blocks[0].srcdata["feat"].float()
        if self.ablation == "without_structure":
            structural = torch.zeros_like(structural)
        h = self.feature_encoder(
            features, structural, use_soft_binning=self.ablation != "without_soft_binning"
        )
        risk = torch.zeros_like(h) if self.ablation == "without_structure" else self.risk_encoder(structural)
        risk_gate = torch.sigmoid(self.input_risk_gate(torch.cat([h, risk], dim=-1)))
        h = self.input_norm(h + risk_gate * risk)
        root_count = blocks[-1].num_dst_nodes()
        root = h[:root_count]
        layer_stats = []
        for layer_index, (layer, block) in enumerate(zip(self.layers, blocks)):
            h, stats = layer(
                block,
                h,
                structural,
                augment,
                perturbation_policy=perturbation_policy,
                perturbation_budget=perturbation_budget,
                perturbation_seed=perturbation_seed + layer_index * 1009,
                collect_diagnostics=collect_diagnostics,
            )
            structural = structural[: block.num_dst_nodes()]
            layer_stats.append(stats)
        local = self.local_fusion(torch.cat([h, root], dim=-1))
        if self.ablation == "without_prototypes":
            output, memory_confidence = local, local.new_tensor(0.0)
        else:
            output, memory_confidence = self.memory(local)
        aggregate_stats = {
            "mean_node_uncertainty": torch.stack(
                [item["mean_node_uncertainty"] for item in layer_stats]
            ).mean(),
            "mean_relation_uncertainty": torch.stack(
                [item["mean_relation_uncertainty"] for item in layer_stats]
            ).mean(),
            "calibration_loss": torch.stack(
                [item["calibration_loss"] for item in layer_stats]
            ).mean(),
            "memory_confidence": memory_confidence,
        }
        if collect_diagnostics:
            aggregate_stats["relation_diagnostics"] = [
                {
                    "layer": index,
                    "uncertainty": item["relation_uncertainty_values"],
                    "target": item["calibration_target_values"],
                }
                for index, item in enumerate(layer_stats)
            ]
        return output, aggregate_stats

    def forward(self, blocks: list[dgl.DGLGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        first, first_stats = self.encode_view(blocks, self.training)
        second, second_stats = self.encode_view(blocks, self.training)
        logits = self.classifier(0.5 * (first + second)).squeeze(-1)
        stats = {
            key: 0.5 * (first_stats[key] + second_stats[key])
            for key in ("mean_node_uncertainty", "mean_relation_uncertainty", "calibration_loss")
        }
        stats["memory_confidence"] = 0.5 * (
            first_stats["memory_confidence"] + second_stats["memory_confidence"]
        )
        return logits, self.projector(first), self.projector(second), stats


if __name__ == "__main__":
    runner.BaseRUDG = RUDG
    runner.main()
