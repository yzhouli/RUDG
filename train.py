"""Train RUDG and select a checkpoint using validation PR-AUC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cli_utils import ROOT, load_config, runner_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("amazon", "yelpchi", "dgraphfin", "tfinance"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed data-split seed.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    config = load_config(args.dataset, args.config)
    output = args.output or ROOT / "outputs" / "validation" / f"rudg_{args.dataset}_seed{args.seed}.json"
    checkpoint = args.checkpoint or ROOT / "outputs" / "checkpoints" / f"rudg_{args.dataset}_seed{args.seed}.state.pt"

    from models.rudg import model, runner

    runner.BaseRUDG = model.RUDG
    sys.argv = [
        "train.py", "--dataset", args.dataset, "--mode", "validation",
        "--device", args.device, "--seed", str(args.seed),
        "--split-seed", str(args.split_seed), "--output", str(output),
        "--state-output", str(checkpoint), *runner_arguments(config),
    ]
    runner.main()


if __name__ == "__main__":
    main()
