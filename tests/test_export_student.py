"""Unit tests for src/models/export_student_gguf.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.export_student_gguf import (  # noqa: E402
    UsageError,
    _GGUF_TYPE_ARRAY,
    _GGUF_TYPE_BOOL,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_INT32,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT32,
    build_output_vmap,
    copy_tokenizer_keys,
    main,
    validate_checkpoint,
)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeField:
    def __init__(self, types: list[int], value: object) -> None:
        self.types = types
        self._value = value

    def contents(self) -> object:
        return self._value


class _FakeReader:
    def __init__(self, fields: dict[str, _FakeField]) -> None:
        self.fields = fields


class _FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def add_string(self, k: str, v: str) -> None:
        self.calls.append(("string", k, v))

    def add_uint32(self, k: str, v: int) -> None:
        self.calls.append(("uint32", k, v))

    def add_bool(self, k: str, v: bool) -> None:
        self.calls.append(("bool", k, v))

    def add_array(self, k: str, v: object) -> None:
        self.calls.append(("array", k, v))


def _make_valid_ckpt(n_vocab_in: int = 8, n_vprime: int = 4, hidden: int = 6) -> dict:
    return {
        "config": {
            "n_vocab_in": n_vocab_in,
            "n_vprime": n_vprime,
            "hidden": hidden,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 4,
            "ffn": 12,
            "rope_theta": 1e6,
            "rms_eps": 1e-6,
            "max_seq": 64,
        },
        "vprime": list(range(n_vprime)),
        "state_dict": {
            "embed_tokens.weight": _FakeTensor((n_vocab_in, hidden)),
            "layers.0.input_layernorm.weight": _FakeTensor((hidden,)),
            "layers.0.self_attn.q_proj.weight": _FakeTensor((8, hidden)),
            "layers.0.self_attn.k_proj.weight": _FakeTensor((4, hidden)),
            "layers.0.self_attn.v_proj.weight": _FakeTensor((4, hidden)),
            "layers.0.self_attn.o_proj.weight": _FakeTensor((hidden, 8)),
            "layers.0.self_attn.q_norm.weight": _FakeTensor((4,)),
            "layers.0.self_attn.k_norm.weight": _FakeTensor((4,)),
            "layers.0.post_attention_layernorm.weight": _FakeTensor((hidden,)),
            "layers.0.mlp.gate_proj.weight": _FakeTensor((12, hidden)),
            "layers.0.mlp.up_proj.weight": _FakeTensor((12, hidden)),
            "layers.0.mlp.down_proj.weight": _FakeTensor((hidden, 12)),
            "norm.weight": _FakeTensor((hidden,)),
            "lm_head.weight": _FakeTensor((n_vprime, hidden)),
        },
    }


class TestExportStudentGGUF(unittest.TestCase):
    def test_help_works_without_gguf_or_torch(self) -> None:
        script = SRC_DIR / "models" / "export_student_gguf.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--ckpt", proc.stdout)
        self.assertIn("--ref-gguf", proc.stdout)
        self.assertIn("--out", proc.stdout)

    def test_missing_inputs_exit_2_without_creating_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "out.gguf"
            rc = main(
                [
                    "--ckpt",
                    str(tmp_path / "missing.pt"),
                    "--ref-gguf",
                    str(tmp_path / "missing.gguf"),
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 2)
            self.assertFalse(out_path.exists())

    def test_validate_checkpoint_accepts_valid(self) -> None:
        ckpt = _make_valid_ckpt()
        cfg, _, vprime = validate_checkpoint(ckpt)
        self.assertEqual(cfg["n_vocab_in"], 8)
        self.assertEqual(vprime, [0, 1, 2, 3])

    def test_validate_checkpoint_rejects_bad_vprime(self) -> None:
        ckpt = _make_valid_ckpt()
        ckpt["vprime"] = [0, 1, 1, 2]
        with self.assertRaises(UsageError):
            validate_checkpoint(ckpt)

        ckpt = _make_valid_ckpt()
        ckpt["vprime"] = [0, 1, 2, 99]
        with self.assertRaises(UsageError):
            validate_checkpoint(ckpt)

        ckpt = _make_valid_ckpt()
        ckpt["vprime"] = [0, 1, 2]
        with self.assertRaises(UsageError):
            validate_checkpoint(ckpt)

    def test_validate_checkpoint_rejects_missing_tensor(self) -> None:
        ckpt = _make_valid_ckpt()
        del ckpt["state_dict"]["lm_head.weight"]
        with self.assertRaises(UsageError):
            validate_checkpoint(ckpt)

    def test_copy_tokenizer_keys_with_fake_reader(self) -> None:
        reader = _FakeReader(
            {
                "general.architecture": _FakeField([_GGUF_TYPE_STRING], "qwen3"),
                "tokenizer.ggml.model": _FakeField([_GGUF_TYPE_STRING], "gpt2"),
                "tokenizer.ggml.tokens": _FakeField(
                    [_GGUF_TYPE_ARRAY, _GGUF_TYPE_STRING], ["a", "b", "c", "d"]
                ),
                "tokenizer.ggml.bos_token_id": _FakeField([_GGUF_TYPE_UINT32], 1),
                "tokenizer.ggml.add_bos_token": _FakeField([_GGUF_TYPE_BOOL], False),
            }
        )
        writer = _FakeWriter()
        copied = copy_tokenizer_keys(reader, writer, n_vocab_in=4)
        self.assertEqual(copied, 4)
        keys = [k for _, k, _ in writer.calls]
        self.assertNotIn("general.architecture", keys)
        self.assertIn("tokenizer.ggml.tokens", keys)
        self.assertIn("tokenizer.ggml.bos_token_id", keys)

        with self.assertRaises(UsageError):
            copy_tokenizer_keys(reader, _FakeWriter(), n_vocab_in=5)

    def test_copy_tokenizer_int32_array_passes_list(self) -> None:
        reader = _FakeReader(
            {
                "tokenizer.ggml.tokens": _FakeField(
                    [_GGUF_TYPE_ARRAY, _GGUF_TYPE_STRING], ["a", "b", "c", "d"]
                ),
                "tokenizer.ggml.token_type": _FakeField(
                    [_GGUF_TYPE_ARRAY, _GGUF_TYPE_INT32], [1, 1, 3, 1]
                ),
            }
        )
        writer = _FakeWriter()
        self.assertEqual(copy_tokenizer_keys(reader, writer, n_vocab_in=4), 2)
        token_type_calls = [
            v for kind, k, v in writer.calls if kind == "array" and k == "tokenizer.ggml.token_type"
        ]
        self.assertEqual(len(token_type_calls), 1)
        self.assertIs(type(token_type_calls[0]), list)
        self.assertEqual(token_type_calls[0], [1, 1, 3, 1])
        self.assertTrue(all(type(x) is int for x in token_type_calls[0]))

    @unittest.skipUnless(HAS_NUMPY, "numpy not installed")
    def test_copy_tokenizer_numeric_arrays_and_vmap(self) -> None:
        reader = _FakeReader(
            {
                "tokenizer.ggml.tokens": _FakeField(
                    [_GGUF_TYPE_ARRAY, _GGUF_TYPE_STRING], ["a", "b", "c", "d"]
                ),
                "tokenizer.ggml.token_type": _FakeField(
                    [_GGUF_TYPE_ARRAY, _GGUF_TYPE_INT32], [1, 1, 3, 1]
                ),
                "tokenizer.ggml.scores": _FakeField(
                    [_GGUF_TYPE_ARRAY, _GGUF_TYPE_FLOAT32], [0.1, 0.2, 0.3, 0.4]
                ),
            }
        )
        writer = _FakeWriter()
        self.assertEqual(copy_tokenizer_keys(reader, writer, n_vocab_in=4), 3)

        vmap = build_output_vmap([3, 1], n_vocab_in=5)
        self.assertEqual(str(vmap.dtype), "int32")
        self.assertEqual(vmap.tolist(), [2, 1, 2, 0, 2])


if __name__ == "__main__":
    unittest.main()
