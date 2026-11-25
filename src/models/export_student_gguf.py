"""Export a trained Student checkpoint (.pt) to a Qwen3 GGUF with V' output head."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any

# GGUF value type IDs (matches gguf.GGUFValueType).
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_REQUIRED_CFG_INT_FIELDS = (
    "n_vocab_in",
    "n_vprime",
    "hidden",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "ffn",
    "max_seq",
)
_REQUIRED_CFG_FLOAT_FIELDS = ("rope_theta", "rms_eps")


class UsageError(ValueError):
    """Raised for invalid CLI arguments, missing inputs, or malformed checkpoints (exit 2)."""


def _import_gguf() -> Any:
    gguf_py = Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp" / "gguf-py"
    if gguf_py.is_dir():
        gguf_py_str = str(gguf_py)
        if gguf_py_str not in sys.path:
            sys.path.insert(0, gguf_py_str)
    import gguf  # type: ignore[import-not-found]

    return gguf


def _tensor_specs(n_layers: int) -> list[tuple[str, str, bool]]:
    """Return (state_dict_key, gguf_name, is_norm) tuples for all model weights."""
    specs: list[tuple[str, str, bool]] = [
        ("embed_tokens.weight", "token_embd.weight", False),
    ]
    for i in range(n_layers):
        specs.extend(
            [
                (f"layers.{i}.input_layernorm.weight", f"blk.{i}.attn_norm.weight", True),
                (f"layers.{i}.self_attn.q_proj.weight", f"blk.{i}.attn_q.weight", False),
                (f"layers.{i}.self_attn.k_proj.weight", f"blk.{i}.attn_k.weight", False),
                (f"layers.{i}.self_attn.v_proj.weight", f"blk.{i}.attn_v.weight", False),
                (f"layers.{i}.self_attn.o_proj.weight", f"blk.{i}.attn_output.weight", False),
                (f"layers.{i}.self_attn.q_norm.weight", f"blk.{i}.attn_q_norm.weight", True),
                (f"layers.{i}.self_attn.k_norm.weight", f"blk.{i}.attn_k_norm.weight", True),
                (
                    f"layers.{i}.post_attention_layernorm.weight",
                    f"blk.{i}.ffn_norm.weight",
                    True,
                ),
                (f"layers.{i}.mlp.gate_proj.weight", f"blk.{i}.ffn_gate.weight", False),
                (f"layers.{i}.mlp.up_proj.weight", f"blk.{i}.ffn_up.weight", False),
                (f"layers.{i}.mlp.down_proj.weight", f"blk.{i}.ffn_down.weight", False),
            ]
        )
    specs.extend(
        [
            ("norm.weight", "output_norm.weight", True),
            ("lm_head.weight", "output.weight", False),
        ]
    )
    return specs


def _expected_tensor_shapes(cfg: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return expected PyTorch tensor shapes for every required state_dict key."""
    hidden = cfg["hidden"]
    q_dim = cfg["n_heads"] * cfg["head_dim"]
    kv_dim = cfg["n_kv_heads"] * cfg["head_dim"]
    ffn = cfg["ffn"]
    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (cfg["n_vocab_in"], hidden),
        "norm.weight": (hidden,),
        "lm_head.weight": (cfg["n_vprime"], hidden),
    }
    for i in range(cfg["n_layers"]):
        shapes.update(
            {
                f"layers.{i}.input_layernorm.weight": (hidden,),
                f"layers.{i}.self_attn.q_proj.weight": (q_dim, hidden),
                f"layers.{i}.self_attn.k_proj.weight": (kv_dim, hidden),
                f"layers.{i}.self_attn.v_proj.weight": (kv_dim, hidden),
                f"layers.{i}.self_attn.o_proj.weight": (hidden, q_dim),
                f"layers.{i}.self_attn.q_norm.weight": (cfg["head_dim"],),
                f"layers.{i}.self_attn.k_norm.weight": (cfg["head_dim"],),
                f"layers.{i}.post_attention_layernorm.weight": (hidden,),
                f"layers.{i}.mlp.gate_proj.weight": (ffn, hidden),
                f"layers.{i}.mlp.up_proj.weight": (ffn, hidden),
                f"layers.{i}.mlp.down_proj.weight": (hidden, ffn),
            }
        )
    return shapes


def _is_floating_point_tensor(tensor: Any) -> bool:
    """Return True if tensor has a floating-point dtype."""
    is_fp = getattr(tensor, "is_floating_point", None)
    if callable(is_fp):
        return bool(is_fp())
    dtype = getattr(tensor, "dtype", None)
    if dtype is not None:
        if hasattr(dtype, "is_floating_point"):
            return bool(dtype.is_floating_point)
        if hasattr(dtype, "kind"):
            return dtype.kind == "f"
    return False


def validate_checkpoint(ckpt: Any) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    """Validate a loaded Student checkpoint and return (config, state_dict, vprime)."""
    if not isinstance(ckpt, dict):
        raise UsageError("Malformed checkpoint: top-level object must be a dict")
    for key in ("config", "state_dict", "vprime"):
        if key not in ckpt:
            raise UsageError(f"Malformed checkpoint: missing {key!r}")

    raw_cfg = ckpt["config"]
    if not isinstance(raw_cfg, dict):
        raise UsageError("Malformed checkpoint: 'config' must be a dict")

    # b10 FFN variants have no llama.cpp graph; only SwiGLU checkpoints can be exported.
    ffn_type = raw_cfg.get("ffn_type", "swiglu")
    if ffn_type != "swiglu":
        raise UsageError(
            f"Unsupported config.ffn_type {ffn_type!r}: only 'swiglu' checkpoints can be "
            "exported to GGUF (llama.cpp has no graph for the b10 FFN variants)"
        )

    cfg: dict[str, Any] = {}
    for field in _REQUIRED_CFG_INT_FIELDS:
        if field not in raw_cfg:
            raise UsageError(f"Malformed checkpoint config: missing {field!r}")
        val = raw_cfg[field]
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise UsageError(
                f"Malformed checkpoint config: {field!r} must be a positive int, got {val!r}"
            )
        cfg[field] = int(val)

    for field in _REQUIRED_CFG_FLOAT_FIELDS:
        if field not in raw_cfg:
            raise UsageError(f"Malformed checkpoint config: missing {field!r}")
        val = raw_cfg[field]
        if (
            not isinstance(val, (int, float))
            or isinstance(val, bool)
            or not math.isfinite(float(val))
            or float(val) <= 0.0
        ):
            raise UsageError(
                f"Malformed checkpoint config: {field!r} must be a positive finite float, got {val!r}"
            )
        cfg[field] = float(val)

    n_vocab_in = cfg["n_vocab_in"]
    n_vprime = cfg["n_vprime"]
    if n_vprime >= n_vocab_in:
        raise UsageError(
            f"Malformed checkpoint config: n_vprime ({n_vprime}) >= n_vocab_in ({n_vocab_in}); "
            "llama.cpp requires n_vprime < n_vocab_in"
        )

    raw_vprime = ckpt["vprime"]
    if not isinstance(raw_vprime, (list, tuple)):
        raise UsageError("Malformed checkpoint: 'vprime' must be a list or tuple")
    vprime: list[int] = []
    for idx, item in enumerate(raw_vprime):
        if not isinstance(item, int) or isinstance(item, bool):
            raise UsageError(f"Malformed vprime[{idx}]: expected int, got {item!r}")
        if item < 0 or item >= n_vocab_in:
            raise UsageError(
                f"Out-of-range vprime[{idx}]={item} for n_vocab_in={n_vocab_in}"
            )
        vprime.append(int(item))

    if len(vprime) != n_vprime:
        raise UsageError(
            f"Checkpoint vprime length ({len(vprime)}) != config.n_vprime ({n_vprime})"
        )
    if len(set(vprime)) != len(vprime):
        raise UsageError("Checkpoint vprime contains duplicate token IDs")

    state_dict = ckpt["state_dict"]
    if not isinstance(state_dict, dict):
        raise UsageError("Malformed checkpoint: 'state_dict' must be a dict")

    expected_shapes = _expected_tensor_shapes(cfg)
    for sd_key, _, _ in _tensor_specs(cfg["n_layers"]):
        if sd_key not in state_dict:
            raise UsageError(f"Malformed checkpoint state_dict: missing {sd_key!r}")
        tensor = state_dict[sd_key]
        if not hasattr(tensor, "shape"):
            raise UsageError(
                f"Malformed checkpoint state_dict: {sd_key!r} has no 'shape' attribute"
            )
        actual_shape = tuple(tensor.shape)
        exp_shape = expected_shapes[sd_key]
        if actual_shape != exp_shape:
            raise UsageError(
                f"Malformed checkpoint state_dict: {sd_key!r} shape {actual_shape} != {exp_shape}"
            )
        if not _is_floating_point_tensor(tensor):
            raise UsageError(
                f"Malformed checkpoint state_dict: {sd_key!r} must be floating-point"
            )

    return cfg, state_dict, vprime


def build_output_vmap(vprime: list[int], n_vocab_in: int) -> Any:
    """Build I32 output_vmap.weight of shape (n_vocab_in,) mapping token ID -> V' row."""
    import numpy as np

    n_vprime = len(vprime)
    vmap = np.full((n_vocab_in,), n_vprime, dtype=np.int32)
    for row_idx, tok_id in enumerate(vprime):
        vmap[tok_id] = row_idx
    return vmap


def extract_field_value(field: Any) -> Any:
    """Extract a Python value from a gguf.ReaderField (or compatible fake)."""
    if hasattr(field, "contents") and callable(field.contents):
        return field.contents()
    if hasattr(field, "value"):
        return field.value

    types = [int(t) for t in field.types]
    if not types:
        raise ValueError("GGUF field has empty types")

    primary = types[0]
    if primary == _GGUF_TYPE_ARRAY:
        if len(types) < 2:
            raise ValueError("GGUF array field missing element type")
        elem = types[1]
        if elem == _GGUF_TYPE_STRING:
            return [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]
        if elem == _GGUF_TYPE_BOOL:
            return [bool(field.parts[idx].tolist()[0]) for idx in field.data]
        return [field.parts[idx].tolist()[0] for idx in field.data]

    raw = field.parts[field.data[-1]]
    if primary == _GGUF_TYPE_STRING:
        return bytes(raw).decode("utf-8")
    if primary == _GGUF_TYPE_BOOL:
        return bool(raw.tolist()[0])
    return raw.tolist()[0]


def add_field_to_writer(writer: Any, key: str, field: Any) -> None:
    """Re-add a GGUFReader field to a GGUFWriter preserving its GGUF value type."""
    types = [int(t) for t in field.types]
    if not types:
        raise ValueError(f"GGUF field {key!r} has empty types")

    val = extract_field_value(field)
    primary = types[0]

    scalar_dispatch = {
        _GGUF_TYPE_UINT8: lambda k, v: writer.add_uint8(k, int(v)),
        _GGUF_TYPE_INT8: lambda k, v: writer.add_int8(k, int(v)),
        _GGUF_TYPE_UINT16: lambda k, v: writer.add_uint16(k, int(v)),
        _GGUF_TYPE_INT16: lambda k, v: writer.add_int16(k, int(v)),
        _GGUF_TYPE_UINT32: lambda k, v: writer.add_uint32(k, int(v)),
        _GGUF_TYPE_INT32: lambda k, v: writer.add_int32(k, int(v)),
        _GGUF_TYPE_FLOAT32: lambda k, v: writer.add_float32(k, float(v)),
        _GGUF_TYPE_BOOL: lambda k, v: writer.add_bool(k, bool(v)),
        _GGUF_TYPE_STRING: lambda k, v: writer.add_string(k, str(v)),
        _GGUF_TYPE_UINT64: lambda k, v: writer.add_uint64(k, int(v)),
        _GGUF_TYPE_INT64: lambda k, v: writer.add_int64(k, int(v)),
        _GGUF_TYPE_FLOAT64: lambda k, v: writer.add_float64(k, float(v)),
    }
    if primary in scalar_dispatch:
        scalar_dispatch[primary](key, val)
        return

    if primary == _GGUF_TYPE_ARRAY:
        if len(types) < 2:
            raise ValueError(f"GGUF array field {key!r} missing element type")
        elem = types[1]
        array_casters: dict[int, type] = {
            _GGUF_TYPE_STRING: str,
            _GGUF_TYPE_BOOL: bool,
            _GGUF_TYPE_UINT8: int,
            _GGUF_TYPE_INT8: int,
            _GGUF_TYPE_UINT16: int,
            _GGUF_TYPE_INT16: int,
            _GGUF_TYPE_UINT32: int,
            _GGUF_TYPE_INT32: int,
            _GGUF_TYPE_UINT64: int,
            _GGUF_TYPE_INT64: int,
            _GGUF_TYPE_FLOAT32: float,
            _GGUF_TYPE_FLOAT64: float,
        }
        caster = array_casters.get(elem)
        if caster is None:
            raise ValueError(f"Unsupported GGUF array element type {elem} for {key!r}")
        writer.add_array(key, [caster(x) for x in val])
        return

    raise ValueError(f"Unsupported GGUF value type {primary} for {key!r}")


def copy_tokenizer_keys(reader: Any, writer: Any, n_vocab_in: int) -> int:
    """Validate reference vocab size and copy all `tokenizer.*` keys to writer."""
    tokens_key = "tokenizer.ggml.tokens"
    if tokens_key not in reader.fields:
        raise UsageError(f"Reference GGUF missing {tokens_key!r}")

    ref_tokens = extract_field_value(reader.fields[tokens_key])
    if len(ref_tokens) != n_vocab_in:
        raise UsageError(
            f"Vocabulary size mismatch: reference GGUF has {len(ref_tokens)} tokens, "
            f"checkpoint config has n_vocab_in={n_vocab_in}"
        )

    copied = 0
    for key, field in reader.fields.items():
        if key.startswith("tokenizer."):
            add_field_to_writer(writer, key, field)
            copied += 1
    return copied


def export_student_gguf(
    ckpt_path: Path,
    ref_gguf_path: Path,
    out_path: Path,
    dtype: str = "f16",
) -> None:
    """Export a Student checkpoint to GGUF via OUT.gguf.tmp -> OUT.gguf."""
    if not ckpt_path.is_file():
        raise UsageError(f"Checkpoint not found: {ckpt_path}")
    if not ref_gguf_path.is_file():
        raise UsageError(f"Reference GGUF not found: {ref_gguf_path}")
    if dtype not in ("f32", "f16"):
        raise UsageError(f"Unsupported dtype {dtype!r}; expected 'f32' or 'f16'")

    import numpy as np
    import torch

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise UsageError(f"Failed to load checkpoint {ckpt_path}: {exc}") from exc
    cfg, state_dict, vprime = validate_checkpoint(ckpt)

    gguf = _import_gguf()
    reader = gguf.GGUFReader(str(ref_gguf_path), "r")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    matrix_dtype = np.float16 if dtype == "f16" else np.float32

    try:
        writer = gguf.GGUFWriter(str(tmp_path), "qwen3")
        writer.add_name(out_path.stem)
        writer.add_uint32("qwen3.context_length", cfg["max_seq"])
        writer.add_uint32("qwen3.embedding_length", cfg["hidden"])
        writer.add_uint32("qwen3.block_count", cfg["n_layers"])
        writer.add_uint32("qwen3.feed_forward_length", cfg["ffn"])
        writer.add_uint32("qwen3.attention.head_count", cfg["n_heads"])
        writer.add_uint32("qwen3.attention.head_count_kv", cfg["n_kv_heads"])
        writer.add_uint32("qwen3.attention.key_length", cfg["head_dim"])
        writer.add_uint32("qwen3.attention.value_length", cfg["head_dim"])
        writer.add_float32("qwen3.rope.freq_base", cfg["rope_theta"])
        writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", cfg["rms_eps"])

        copy_tokenizer_keys(reader, writer, cfg["n_vocab_in"])

        for sd_key, gguf_name, is_norm in _tensor_specs(cfg["n_layers"]):
            target_dtype = np.float32 if is_norm else matrix_dtype
            arr = state_dict[sd_key].detach().cpu().float().numpy().astype(
                target_dtype, copy=False
            )
            writer.add_tensor(gguf_name, arr)

        writer.add_tensor("output_vmap.weight", build_output_vmap(vprime, cfg["n_vocab_in"]))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained Student checkpoint to a Qwen3 GGUF with V' head."
    )
    parser.add_argument("--ckpt", required=True, help="Path to student checkpoint (.pt)")
    parser.add_argument(
        "--ref-gguf", required=True, help="Path to reference Qwen3 GGUF for tokenizer"
    )
    parser.add_argument("--out", required=True, help="Output GGUF path")
    parser.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f16",
        help="Matrix weight dtype (default: f16; norms are always f32)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        export_student_gguf(
            ckpt_path=Path(args.ckpt),
            ref_gguf_path=Path(args.ref_gguf),
            out_path=Path(args.out),
            dtype=args.dtype,
        )
        return 0
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
