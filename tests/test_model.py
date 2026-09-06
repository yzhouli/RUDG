"""Small CPU-only checks that do not require downloading a benchmark dataset."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / ".cache" / "dgl"))

import dgl
import torch


sys.path.insert(0, str(ROOT))

from models.rudg.model import RUDG  # noqa: E402
from models.rudg.runner import attach_structural_features  # noqa: E402


class RUDGModelTest(unittest.TestCase):
    def test_sampled_forward_is_finite(self) -> None:
        source = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6])
        target = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 0])
        graph = dgl.add_self_loop(dgl.graph((source, target), num_nodes=8))
        graph.ndata["feat"] = torch.randn(8, 5)
        graph.ndata["label"] = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1])
        attach_structural_features(graph)

        sampler = dgl.dataloading.MultiLayerNeighborSampler([3, 2])
        loader = dgl.dataloading.DataLoader(
            graph, torch.tensor([0, 1, 2, 3]), sampler,
            batch_size=4, shuffle=False, num_workers=0,
        )
        _, output_nodes, blocks = next(iter(loader))
        model = RUDG(
            input_dim=5, hidden=8, dropout=0.0, max_aug_drop=0.1,
            feature_bins=4, memory_tokens=4, attention_heads=2, layers=2,
        )
        model.eval()
        with torch.no_grad():
            logits, first_view, second_view, statistics = model(blocks)

        self.assertEqual(tuple(logits.shape), (len(output_nodes),))
        self.assertEqual(tuple(first_view.shape), (len(output_nodes), 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(second_view).all())
        self.assertTrue(torch.isfinite(statistics["calibration_loss"]))


if __name__ == "__main__":
    unittest.main()
