"""Shared configuration handling for the train and test entry points."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

FLAG_NAMES = {
    "hidden": "hidden",
    "layers": "layers",
    "attention_heads": "attention-heads",
    "feature_bins": "feature-bins",
    "memory_tokens": "memory-tokens",
    "max_aug_drop": "max-aug-drop",
    "contrast_weight": "contrast-weight",
    "calibration_weight": "calibration-weight",
    "calibration_representation_mix": "calibration-representation-mix",
    "reliability_floor": "reliability-floor",
    "uncertainty_penalty_init": "uncertainty-penalty-init",
    "tau": "tau",
    "dropout": "dropout",
    "focal_gamma": "focal-gamma",
    "pos_weight_scale": "pos-weight-scale",
    "lr": "lr",
    "weight_decay": "weight-decay",
    "max_epochs": "max-epochs",
    "patience": "patience",
    "contrast_batch": "contrast-batch",
    "batch_size": "batch-size",
    "eval_batch_size": "eval-batch-size",
}


def load_config(dataset: str, path: Path | None) -> dict:
    config_path = path or ROOT / "configs" / f"{dataset}.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    if config.get("dataset") != dataset:
        raise ValueError(
            f"configuration dataset is {config.get('dataset')!r}, expected {dataset!r}"
        )
    if config.get("contrast_weight") != config.get("calibration_weight"):
        raise ValueError("RUDG requires contrast_weight == calibration_weight")
    if len(config.get("fanouts", [])) != int(config["layers"]):
        raise ValueError("the number of fanouts must equal the number of layers")
    return config


def runner_arguments(config: dict) -> list[str]:
    arguments: list[str] = []
    for key, flag in FLAG_NAMES.items():
        arguments.extend([f"--{flag}", str(config[key])])
    arguments.extend(["--fanouts", ",".join(str(value) for value in config["fanouts"])])
    return arguments
