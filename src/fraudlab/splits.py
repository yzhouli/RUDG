"""Auditable split manifests shared by all graph-fraud models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_80_10_10(
    eligible: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic stratified node indices, sorted within each split."""
    eligible = np.asarray(eligible, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1)
    train_idx, holdout_idx = train_test_split(
        eligible,
        train_size=0.8,
        stratify=labels[eligible],
        random_state=seed,
        shuffle=True,
    )
    valid_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.5,
        stratify=labels[holdout_idx],
        random_state=seed,
        shuffle=True,
    )
    return tuple(np.sort(x.astype(np.int64, copy=False)) for x in (train_idx, valid_idx, test_idx))


def validate_split(
    eligible: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    eligible = np.sort(np.asarray(eligible, dtype=np.int64))
    parts = [np.asarray(x, dtype=np.int64) for x in (train_idx, valid_idx, test_idx)]
    for name, part in zip(("train", "valid", "test"), parts):
        if len(part) != len(np.unique(part)):
            raise ValueError(f"duplicate node in {name} split")
        if not np.all(np.isin(part, eligible)):
            raise ValueError(f"ineligible node in {name} split")
        if len(np.unique(labels[part])) < 2:
            raise ValueError(f"{name} split does not contain both classes")
    if np.intersect1d(parts[0], parts[1]).size or np.intersect1d(parts[0], parts[2]).size or np.intersect1d(parts[1], parts[2]).size:
        raise ValueError("split overlap detected")
    if not np.array_equal(np.sort(np.concatenate(parts)), eligible):
        raise ValueError("splits do not exactly cover eligible nodes")


def _array_digest(hasher: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    value = np.ascontiguousarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    hasher.update(value.tobytes())


def split_digest(dataset: str, seed: int, arrays: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    hasher.update(dataset.encode("utf-8"))
    hasher.update(np.asarray([seed], dtype=np.int64).tobytes())
    for name in ("eligible_idx", "train_idx", "valid_idx", "test_idx"):
        _array_digest(hasher, name, arrays[name])
    return hasher.hexdigest().upper()


def class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def save_split_manifest(
    output_dir: Path,
    dataset: str,
    labels: np.ndarray,
    eligible: np.ndarray,
    seed: int,
    source_file: str,
    source_sha256: str,
    notes: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_idx, valid_idx, test_idx = stratified_80_10_10(eligible, labels, seed)
    validate_split(eligible, labels, train_idx, valid_idx, test_idx)
    arrays = {
        "eligible_idx": np.sort(np.asarray(eligible, dtype=np.int64)),
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
    }
    digest = split_digest(dataset, seed, arrays)
    stem = f"{dataset}_stratified_80_10_10_seed{seed}"
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
    metadata = {
        "dataset": dataset,
        "protocol": "transductive_stratified_random_80_10_10",
        "seed": seed,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "split_sha256": digest,
        "eligible_nodes": int(len(eligible)),
        "counts": {
            "train": {"nodes": int(len(train_idx)), "classes": class_counts(labels, train_idx)},
            "valid": {"nodes": int(len(valid_idx)), "classes": class_counts(labels, valid_idx)},
            "test": {"nodes": int(len(test_idx)), "classes": class_counts(labels, test_idx)},
        },
        "notes": notes,
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def load_split_manifest(path: Path, dataset: str | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name], dtype=np.int64) for name in source.files}
    required = {"eligible_idx", "train_idx", "valid_idx", "test_idx"}
    if set(arrays) != required:
        raise ValueError(f"unexpected split fields: {sorted(arrays)}")
    if dataset is not None and seed is not None:
        expected = path.with_suffix(".json")
        metadata = json.loads(expected.read_text(encoding="utf-8"))
        digest = split_digest(dataset, seed, arrays)
        if digest != metadata["split_sha256"]:
            raise ValueError("split hash mismatch")
    return arrays
