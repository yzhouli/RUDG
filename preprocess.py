"""Convert the four supported raw datasets into leakage-safe DGL graph caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / ".cache" / "dgl"))

import dgl
import numpy as np
import torch
from scipy.io import loadmat


sys.path.insert(0, str(ROOT / "src"))

from fraudlab.splits import save_split_manifest  # noqa: E402


DEFAULT_RAW_PATHS = {
    "amazon": ROOT / "data" / "raw" / "Amazon.mat",
    "yelpchi": ROOT / "data" / "raw" / "YelpChi.mat",
    "dgraphfin": ROOT / "data" / "raw" / "dgraphfin.npz",
    "tfinance": ROOT / "data" / "raw" / "tfinance",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sparse_or_dense(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def load_mat_graph(path: Path, dataset: str):
    raw = loadmat(path)
    required = {"label", "features", "homo"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"{path} is missing MATLAB fields: {sorted(missing)}")
    labels = np.asarray(raw["label"], dtype=np.int64).reshape(-1)
    features = sparse_or_dense(raw["features"])
    adjacency = raw["homo"].tocoo()
    graph = dgl.graph(
        (
            torch.from_numpy(adjacency.row.astype(np.int64, copy=False)),
            torch.from_numpy(adjacency.col.astype(np.int64, copy=False)),
        ),
        num_nodes=len(labels),
    )
    eligible = np.arange(len(labels), dtype=np.int64)
    notes = "all binary-labeled nodes are eligible"
    if dataset == "amazon":
        eligible = eligible[3305:]
        notes = "nodes 0--3304 are excluded from supervision following the standard Amazon protocol"
    return graph, features, labels, eligible, notes


def load_dgraphfin(path: Path):
    with np.load(path, allow_pickle=False) as raw:
        features = np.asarray(raw["x"], dtype=np.float32)
        labels = np.asarray(raw["y"], dtype=np.int64).reshape(-1)
        edges = np.asarray(raw["edge_index"], dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("edge_index must be a two-dimensional array")
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        source, target = edges[0], edges[1]
    elif edges.shape[1] == 2:
        source, target = edges[:, 0], edges[:, 1]
    else:
        raise ValueError("edge_index must have shape [2, E] or [E, 2]")
    graph = dgl.graph((torch.from_numpy(source), torch.from_numpy(target)), num_nodes=len(labels))
    eligible = np.flatnonzero((labels == 0) | (labels == 1)).astype(np.int64)
    notes = "classes 0/1 are supervised; classes 2/3 are retained only as graph context"
    return graph, features, labels, eligible, notes


def load_tfinance(path: Path):
    graph = dgl.load_graphs(str(path))[0][0]
    feature_key = "feature" if "feature" in graph.ndata else "feat"
    if feature_key not in graph.ndata or "label" not in graph.ndata:
        raise KeyError("T-Finance graph must contain node features and labels")
    features = graph.ndata[feature_key].cpu().numpy().astype(np.float32)
    raw_labels = graph.ndata["label"].cpu().numpy()
    labels = (raw_labels[:, 1] if raw_labels.ndim == 2 else raw_labels).astype(np.int64)
    eligible = np.flatnonzero((labels == 0) | (labels == 1)).astype(np.int64)
    notes = "all binary-labeled nodes in the official DGL graph are eligible"
    graph.ndata.clear()
    return graph, features, labels, eligible, notes


def normalize_from_training(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    training = features[train_idx]
    mean = training.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = training.std(axis=0, dtype=np.float64).astype(np.float32)
    constant = std < 1e-6
    std[constant] = 1.0
    normalized = (features - mean) / std
    np.nan_to_num(normalized, copy=False)
    return normalized.astype(np.float32, copy=False), {
        "method": "training-only z-score normalization",
        "constant_columns": int(constant.sum()),
    }


def boolean_mask(size: int, indices: np.ndarray) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.from_numpy(indices)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DEFAULT_RAW_PATHS))
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_path = (args.raw_path or DEFAULT_RAW_PATHS[args.dataset]).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw dataset not found: {raw_path}")

    if args.dataset in {"amazon", "yelpchi"}:
        graph, features, labels, eligible, notes = load_mat_graph(raw_path, args.dataset)
    elif args.dataset == "dgraphfin":
        graph, features, labels, eligible, notes = load_dgraphfin(raw_path)
    else:
        graph, features, labels, eligible, notes = load_tfinance(raw_path)

    source_hash = sha256(raw_path)
    split_dir = ROOT / "data" / "splits"
    metadata = save_split_manifest(
        split_dir, args.dataset, labels, eligible, args.seed,
        str(raw_path.relative_to(ROOT) if raw_path.is_relative_to(ROOT) else raw_path),
        source_hash, notes,
    )
    split_path = split_dir / f"{args.dataset}_stratified_80_10_10_seed{args.seed}.npz"
    with np.load(split_path, allow_pickle=False) as split:
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        valid_idx = np.asarray(split["valid_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)

    features, normalization = normalize_from_training(features, train_idx)
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = torch.from_numpy(features)
    graph.ndata["label"] = torch.from_numpy(labels)
    graph.ndata["train_mask"] = boolean_mask(len(labels), train_idx)
    graph.ndata["val_mask"] = boolean_mask(len(labels), valid_idx)
    graph.ndata["test_mask"] = boolean_mask(len(labels), test_idx)

    cache_name = "gaap_unified" if args.dataset == "dgraphfin" else "gpa_unified"
    cache_dir = ROOT / "data" / "processed" / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{args.dataset}_seed{args.seed}_{metadata['split_sha256'][:12]}.dgl"
    if cache_path.exists() and not args.force:
        raise FileExistsError(f"processed graph already exists: {cache_path}; pass --force to overwrite")
    dgl.save_graphs(str(cache_path), graph)
    report = {
        "dataset": args.dataset,
        "raw_path": str(raw_path),
        "raw_sha256": source_hash,
        "processed_graph": str(cache_path.relative_to(ROOT)),
        "nodes": graph.num_nodes(),
        "directed_edges_after_bidirection": graph.num_edges(),
        "features": int(features.shape[1]),
        "normalization": normalization,
        "split_sha256": metadata["split_sha256"],
    }
    report_path = cache_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
