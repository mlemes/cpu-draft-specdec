"""Unit tests for step_backward and TeacherBatcher seq_len validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _write_dummy_shard_dir(root: Path, seq_len: int, n_seq: int = 2, k: int = 2) -> None:
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    name = "shard_000"
    np.zeros((n_seq, seq_len), dtype="<u4").tofile(root / f"{name}.tokens.u32")
    np.zeros((n_seq, seq_len, k), dtype="<u4").tofile(root / f"{name}.ids.u32")
    np.zeros((n_seq, seq_len, k), dtype="<f2").tofile(root / f"{name}.logp.f16")
    index = {
        "seq_len": seq_len,
        "k": k,
        "n_vocab": 16,
        "shards": [{"name": name, "n_seq": n_seq}],
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY, "torch and numpy required")
class TestStepBackwardAndBatcher(unittest.TestCase):
    def test_step_backward_matches_pooled_kd_and_hard(self) -> None:
        import torch
        import torch.nn as nn

        from train.losses import hard_ce, kd_forward_kl
        from train.train import step_backward

        torch.manual_seed(0)
        model_a = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 4, bias=False))
        model_b = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 4, bias=False))
        model_b.load_state_dict(model_a.state_dict())

        mb1 = {
            "tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "topk_vidx": torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 3]]], dtype=torch.long),
            "topk_p": torch.full((1, 4, 2), 0.4, dtype=torch.float32),
            "loss_mask": torch.tensor([[True, False, False, False]], dtype=torch.bool),
        }
        mb2 = {
            "tokens": torch.tensor([[2, 3, 1, 0]], dtype=torch.long),
            "topk_vidx": torch.tensor([[[0, 2], [1, 3], [0, 1], [2, 3]]], dtype=torch.long),
            "topk_p": torch.full((1, 4, 2), 0.35, dtype=torch.float32),
            "loss_mask": torch.tensor([[True, True, True, False]], dtype=torch.bool),
        }

        # KD pooled check
        model_a.zero_grad()
        loss_a = step_backward(model_a, [mb1, mb2], "kd")

        model_b.zero_grad()
        cat_tokens = torch.cat([mb1["tokens"], mb2["tokens"]], dim=0)
        cat_vidx = torch.cat([mb1["topk_vidx"], mb2["topk_vidx"]], dim=0)
        cat_p = torch.cat([mb1["topk_p"], mb2["topk_p"]], dim=0)
        cat_mask = torch.cat([mb1["loss_mask"], mb2["loss_mask"]], dim=0)
        loss_b = kd_forward_kl(model_b(cat_tokens), cat_vidx, cat_p, cat_mask)
        loss_b.backward()

        self.assertAlmostEqual(loss_a, float(loss_b.item()), places=5)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=1e-6))

        # Hard CE pooled check (token 4 is OOV -> -1 in vmap_t)
        vmap_t = torch.full((16,), -1, dtype=torch.long)
        vmap_t[0] = 0
        vmap_t[1] = 1
        vmap_t[2] = 2
        vmap_t[3] = 3

        model_a.zero_grad()
        hloss_a = step_backward(model_a, [mb1, mb2], "hard", vmap_t=vmap_t)

        model_b.zero_grad()
        hloss_b = hard_ce(model_b(cat_tokens), cat_tokens, vmap_t, cat_mask)
        hloss_b.backward()

        self.assertAlmostEqual(hloss_a, float(hloss_b.item()), places=5)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=1e-6))

    def test_step_backward_zero_positions(self) -> None:
        import torch
        import torch.nn as nn

        from train.train import step_backward

        model = nn.Linear(4, 4)
        mb = {
            "tokens": torch.zeros((1, 4), dtype=torch.long),
            "topk_vidx": torch.zeros((1, 4, 2), dtype=torch.long),
            "topk_p": torch.full((1, 4, 2), 0.4, dtype=torch.float32),
            "loss_mask": torch.zeros((1, 4), dtype=torch.bool),
        }
        loss = step_backward(model, [mb], "kd")
        self.assertEqual(loss, 0.0)
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_teacher_batcher_rejects_mixed_seq_len(self) -> None:
        import numpy as np

        from train.data import TeacherBatcher

        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "d1"
            d2 = Path(tmp) / "d2"
            _write_dummy_shard_dir(d1, seq_len=8)
            _write_dummy_shard_dir(d2, seq_len=16)
            vmap = np.arange(16, dtype=np.int32)
            with self.assertRaises(ValueError):
                TeacherBatcher([(str(d1), 0.5), (str(d2), 0.5)], vmap, batch_seqs=2, seed=0)


if __name__ == "__main__":
    unittest.main()
