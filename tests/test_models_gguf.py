"""Tests for src/models/gguf_io.py and src/models/random_gguf.py."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from models.gguf_io import read_gguf, tensor_nbytes
from models.random_gguf import generate_random_gguf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        check=False,
    )


class TestTensorNbytes(unittest.TestCase):
    def test_known_types(self) -> None:
        self.assertEqual(tensor_nbytes((64,), "F32"), 256)
        self.assertEqual(tensor_nbytes((64, 2), "F16"), 256)
        self.assertEqual(tensor_nbytes((64, 3), "Q8_0"), 6 * 34)
        self.assertEqual(tensor_nbytes((64, 3), "Q4_0"), 6 * 18)

    def test_invalid_quant_ne0_raises(self) -> None:
        with self.assertRaises(ValueError):
            tensor_nbytes((48, 4), "Q8_0")
        with self.assertRaises(ValueError):
            tensor_nbytes((33,), "Q4_0")
        with self.assertRaises(ValueError):
            tensor_nbytes((32,), "UNKNOWN")


class TestGGUFRoundTrip(unittest.TestCase):
    def test_qwen3_roundtrip_and_determinism(self) -> None:
        cfg = {
            "arch": "qwen3",
            "hidden": 64,
            "layers": 1,
            "ffn": 128,
            "heads": 2,
            "kv_heads": 1,
            "head_dim": 32,
            "vocab": 260,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_a = Path(tmp) / "a.gguf"
            out_b = Path(tmp) / "b.gguf"
            out_c = Path(tmp) / "c.gguf"

            generate_random_gguf(cfg, "q8_0", 42, out_a)
            generate_random_gguf(cfg, "q8_0", 42, out_b)
            generate_random_gguf(cfg, "q8_0", 99, out_c)

            bytes_a = out_a.read_bytes()
            self.assertEqual(bytes_a, out_b.read_bytes())
            self.assertNotEqual(bytes_a, out_c.read_bytes())

            gguf = read_gguf(out_a)
            self.assertEqual(gguf.version, 3)
            self.assertEqual(gguf.alignment, 32)
            self.assertEqual(gguf.data_offset % 32, 0)
            self.assertEqual(gguf.metadata["general.architecture"], "qwen3")
            self.assertEqual(gguf.metadata["qwen3.block_count"], 1)
            self.assertEqual(gguf.metadata["qwen3.context_length"], 4096)
            self.assertEqual(len(gguf.metadata["tokenizer.ggml.tokens"]), 260)  # type: ignore[arg-type]

            by_name = {t.name: t for t in gguf.tensors}
            self.assertEqual(by_name["token_embd.weight"].shape, (64, 260))
            self.assertEqual(by_name["token_embd.weight"].ggml_type, "Q8_0")
            self.assertEqual(
                by_name["token_embd.weight"].nbytes,
                tensor_nbytes((64, 260), "Q8_0"),
            )
            self.assertEqual(by_name["blk.0.ffn_down.weight"].shape, (128, 64))
            for t in gguf.tensors:
                self.assertGreaterEqual(t.offset, gguf.data_offset)
                self.assertEqual(t.offset % 32, 0)

            # Check F32 norm weights are 1.0 and Q8_0 scales are finite non-zero.
            norm_info = by_name["output_norm.weight"]
            norm_bytes = bytes_a[norm_info.offset : norm_info.offset + norm_info.nbytes]
            self.assertEqual(struct.unpack("<64f", norm_bytes), (1.0,) * 64)

            emb_info = by_name["token_embd.weight"]
            emb_bytes = bytes_a[emb_info.offset : emb_info.offset + emb_info.nbytes]
            for blk_off in range(0, len(emb_bytes), 34):
                (scale,) = struct.unpack_from("<e", emb_bytes, blk_off)
                self.assertTrue(math.isfinite(scale) and scale != 0.0)

    def test_qwen3moe_q4_0_roundtrip(self) -> None:
        cfg = {
            "arch": "qwen3moe",
            "hidden": 64,
            "layers": 1,
            "heads": 2,
            "kv_heads": 1,
            "head_dim": 32,
            "vocab": 259,
            "experts": 4,
            "experts_used": 2,
            "expert_ffn": 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "moe.gguf"
            generate_random_gguf(cfg, "q4_0", 7, out)
            gguf = read_gguf(out)
            self.assertEqual(gguf.metadata["general.architecture"], "qwen3moe")
            self.assertEqual(gguf.metadata["qwen3moe.expert_count"], 4)
            self.assertNotIn("qwen3moe.feed_forward_length", gguf.metadata)

            by_name = {t.name: t for t in gguf.tensors}
            self.assertEqual(by_name["blk.0.ffn_gate_inp.weight"].ggml_type, "F32")
            self.assertEqual(by_name["blk.0.ffn_gate_inp.weight"].shape, (64, 4))
            self.assertEqual(by_name["blk.0.ffn_gate_exps.weight"].shape, (64, 64, 4))
            self.assertEqual(by_name["blk.0.ffn_gate_exps.weight"].ggml_type, "Q4_0")

    def test_bad_magic_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.gguf"
            bad.write_bytes(b"NOPE" + struct.pack("<IQQ", 3, 0, 0))
            with self.assertRaises(ValueError):
                read_gguf(bad)

            bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 2, 0, 0))
            with self.assertRaises(ValueError):
                read_gguf(bad)


class TestCLIs(unittest.TestCase):
    def test_inspect_cli(self) -> None:
        cfg = {
            "arch": "qwen3",
            "hidden": 32,
            "layers": 1,
            "ffn": 64,
            "heads": 1,
            "kv_heads": 1,
            "head_dim": 32,
            "vocab": 259,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "small.gguf"
            generate_random_gguf(cfg, "q8_0", 1, out)

            res = _run_module("models.gguf_io", "inspect", str(out))
            self.assertEqual(res.returncode, 0, res.stderr)
            payload = json.loads(res.stdout)
            self.assertEqual(payload["version"], 3)
            self.assertEqual(payload["metadata"]["tokenizer.ggml.tokens"], {"array_len": 259})
            self.assertEqual(
                payload["total_tensor_bytes"],
                sum(t["nbytes"] for t in payload["tensors"]),
            )

            missing = _run_module("models.gguf_io", "inspect", str(Path(tmp) / "nope.gguf"))
            self.assertEqual(missing.returncode, 2)
            self.assertTrue(missing.stderr.strip())

    def test_random_gguf_invalid_configs_exit_2_no_file(self) -> None:
        bad_configs = [
            {"arch": "llama", "hidden": 32, "layers": 1, "ffn": 64, "heads": 1, "kv_heads": 1, "head_dim": 32, "vocab": 259},
            {"arch": "qwen3", "hidden": 32, "layers": 1, "heads": 1, "kv_heads": 1, "head_dim": 32, "vocab": 259},
            {"arch": "qwen3", "hidden": 32, "layers": 1, "ffn": 64, "heads": 1, "kv_heads": 1, "head_dim": 32, "vocab": 258},
            {"arch": "qwen3", "hidden": 48, "layers": 1, "ffn": 64, "heads": 1, "kv_heads": 1, "head_dim": 32, "vocab": 259},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for idx, cfg in enumerate(bad_configs):
                cfg_path = Path(tmp) / f"bad_{idx}.json"
                out_path = Path(tmp) / f"out_{idx}.gguf"
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
                res = _run_module(
                    "models.random_gguf",
                    "--config",
                    str(cfg_path),
                    "--quant",
                    "q8_0",
                    "--seed",
                    "1",
                    "--out",
                    str(out_path),
                )
                self.assertEqual(res.returncode, 2, f"config {idx} should fail")
                self.assertTrue(res.stderr.strip())
                self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
