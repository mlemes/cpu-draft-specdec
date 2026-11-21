"""Random-weight Qwen3 / Qwen3-MoE GGUF generator (Python stdlib only)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import struct
import sys
from typing import BinaryIO, Sequence

from models.gguf_io import (
    DEFAULT_ALIGNMENT,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_INT32,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    TensorProducer,
    tensor_nbytes,
    write_gguf,
)

_QUANT_LAYOUT: dict[str, tuple[int, int]] = {
    "Q8_0": (32, 34),
    "Q4_0": (32, 18),
}

# Fixed finite non-zero fp16 scale (0.015625 = 2^-6, exact in binary16).
_FP16_SCALE_B0, _FP16_SCALE_B1 = struct.pack("<e", 0.015625)
_F32_ONE = struct.pack("<f", 1.0)
_CHUNK_BLOCKS = 1_048_576


def _require_pos_int(cfg: dict[str, object], key: str, *, minimum: int = 1) -> int:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key!r}")
    val = cfg[key]
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"Config key {key!r} must be an integer, got {val!r}")
    if val < minimum:
        raise ValueError(f"Config key {key!r} must be >= {minimum}, got {val}")
    return val


def validate_config(cfg: object, quant: str) -> dict[str, object]:
    """Validate model config and return normalized dict; raise ValueError on error."""
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a JSON object")

    arch = cfg.get("arch")
    if arch not in ("qwen3", "qwen3moe"):
        raise ValueError(f"Unsupported arch {arch!r}; expected 'qwen3' or 'qwen3moe'")

    quant_type = quant.upper()
    if quant_type not in _QUANT_LAYOUT:
        raise ValueError(f"Unsupported quant {quant!r}; expected 'q8_0' or 'q4_0'")

    hidden = _require_pos_int(cfg, "hidden")
    layers = _require_pos_int(cfg, "layers")
    heads = _require_pos_int(cfg, "heads")
    kv_heads = _require_pos_int(cfg, "kv_heads")
    head_dim = _require_pos_int(cfg, "head_dim")
    vocab = _require_pos_int(cfg, "vocab", minimum=259)

    if "context" in cfg:
        context = _require_pos_int(cfg, "context")
    else:
        context = 4096

    norm: dict[str, object] = {
        "arch": arch,
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "context": context,
    }

    if arch == "qwen3":
        norm["ffn"] = _require_pos_int(cfg, "ffn")
    else:
        norm["experts"] = _require_pos_int(cfg, "experts")
        norm["experts_used"] = _require_pos_int(cfg, "experts_used")
        norm["expert_ffn"] = _require_pos_int(cfg, "expert_ffn")

    # Verify all tensor shapes (in particular quantized ne0 % 32 == 0).
    for _, shape, ttype in _tensor_specs(norm, quant_type):
        tensor_nbytes(shape, ttype)

    return norm


def _tensor_specs(
    cfg: dict[str, object], quant_type: str
) -> list[tuple[str, tuple[int, ...], str]]:
    arch = str(cfg["arch"])
    hidden = int(cfg["hidden"])  # type: ignore[arg-type]
    layers = int(cfg["layers"])  # type: ignore[arg-type]
    heads = int(cfg["heads"])  # type: ignore[arg-type]
    kv_heads = int(cfg["kv_heads"])  # type: ignore[arg-type]
    head_dim = int(cfg["head_dim"])  # type: ignore[arg-type]
    vocab = int(cfg["vocab"])  # type: ignore[arg-type]
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim

    specs: list[tuple[str, tuple[int, ...], str]] = [
        ("token_embd.weight", (hidden, vocab), quant_type),
        ("output_norm.weight", (hidden,), "F32"),
        ("output.weight", (hidden, vocab), quant_type),
    ]

    for i in range(layers):
        specs.extend(
            [
                (f"blk.{i}.attn_norm.weight", (hidden,), "F32"),
                (f"blk.{i}.attn_q.weight", (hidden, q_dim), quant_type),
                (f"blk.{i}.attn_k.weight", (hidden, kv_dim), quant_type),
                (f"blk.{i}.attn_v.weight", (hidden, kv_dim), quant_type),
                (f"blk.{i}.attn_output.weight", (q_dim, hidden), quant_type),
                (f"blk.{i}.attn_q_norm.weight", (head_dim,), "F32"),
                (f"blk.{i}.attn_k_norm.weight", (head_dim,), "F32"),
                (f"blk.{i}.ffn_norm.weight", (hidden,), "F32"),
            ]
        )
        if arch == "qwen3":
            ffn = int(cfg["ffn"])  # type: ignore[arg-type]
            specs.extend(
                [
                    (f"blk.{i}.ffn_gate.weight", (hidden, ffn), quant_type),
                    (f"blk.{i}.ffn_up.weight", (hidden, ffn), quant_type),
                    (f"blk.{i}.ffn_down.weight", (ffn, hidden), quant_type),
                ]
            )
        else:
            experts = int(cfg["experts"])  # type: ignore[arg-type]
            expert_ffn = int(cfg["expert_ffn"])  # type: ignore[arg-type]
            specs.extend(
                [
                    (f"blk.{i}.ffn_gate_inp.weight", (hidden, experts), "F32"),
                    (
                        f"blk.{i}.ffn_gate_exps.weight",
                        (hidden, expert_ffn, experts),
                        quant_type,
                    ),
                    (
                        f"blk.{i}.ffn_up_exps.weight",
                        (hidden, expert_ffn, experts),
                        quant_type,
                    ),
                    (
                        f"blk.{i}.ffn_down_exps.weight",
                        (expert_ffn, hidden, experts),
                        quant_type,
                    ),
                ]
            )

    return specs


def _build_metadata(cfg: dict[str, object]) -> list[tuple[str, int, object]]:
    arch = str(cfg["arch"])
    vocab = int(cfg["vocab"])  # type: ignore[arg-type]

    tokens = ["<unk>", "<s>", "</s>"]
    tokens.extend(f"<0x{b:02X}>" for b in range(256))
    tokens.extend(f"\u2581t{tid}" for tid in range(259, vocab))

    scores = [0.0] * vocab
    token_types = [2, 3, 3] + [6] * 256 + [1] * (vocab - 259)

    meta: list[tuple[str, int, object]] = [
        ("general.architecture", GGUF_TYPE_STRING, arch),
        ("general.name", GGUF_TYPE_STRING, f"random-{arch}"),
        ("general.alignment", GGUF_TYPE_UINT32, DEFAULT_ALIGNMENT),
        (f"{arch}.block_count", GGUF_TYPE_UINT32, cfg["layers"]),
        (f"{arch}.context_length", GGUF_TYPE_UINT32, cfg["context"]),
        (f"{arch}.embedding_length", GGUF_TYPE_UINT32, cfg["hidden"]),
    ]

    if arch == "qwen3":
        meta.append((f"{arch}.feed_forward_length", GGUF_TYPE_UINT32, cfg["ffn"]))

    meta.extend(
        [
            (f"{arch}.attention.head_count", GGUF_TYPE_UINT32, cfg["heads"]),
            (f"{arch}.attention.head_count_kv", GGUF_TYPE_UINT32, cfg["kv_heads"]),
            (f"{arch}.attention.key_length", GGUF_TYPE_UINT32, cfg["head_dim"]),
            (f"{arch}.attention.value_length", GGUF_TYPE_UINT32, cfg["head_dim"]),
        ]
    )

    if arch == "qwen3moe":
        meta.extend(
            [
                (f"{arch}.expert_count", GGUF_TYPE_UINT32, cfg["experts"]),
                (f"{arch}.expert_used_count", GGUF_TYPE_UINT32, cfg["experts_used"]),
                (
                    f"{arch}.expert_feed_forward_length",
                    GGUF_TYPE_UINT32,
                    cfg["expert_ffn"],
                ),
            ]
        )

    meta.extend(
        [
            (f"{arch}.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, 1e-6),
            (f"{arch}.rope.freq_base", GGUF_TYPE_FLOAT32, 1000000.0),
            ("tokenizer.ggml.model", GGUF_TYPE_STRING, "llama"),
            ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens)),
            ("tokenizer.ggml.scores", GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, scores)),
            ("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, token_types)),
            ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, 1),
            ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, 2),
            ("tokenizer.ggml.unknown_token_id", GGUF_TYPE_UINT32, 0),
        ]
    )
    return meta


def _make_quant_producer(
    n_elems: int, quant_type: str, rng: random.Random
) -> TensorProducer:
    block_elems, block_size = _QUANT_LAYOUT[quant_type]
    total_blocks = n_elems // block_elems

    def _produce(f: BinaryIO) -> None:
        remaining = total_blocks
        while remaining > 0:
            n_blocks = min(remaining, _CHUNK_BLOCKS)
            buf = bytearray(rng.randbytes(n_blocks * block_size))
            buf[0::block_size] = bytes([_FP16_SCALE_B0]) * n_blocks
            buf[1::block_size] = bytes([_FP16_SCALE_B1]) * n_blocks
            f.write(buf)
            remaining -= n_blocks

    return _produce


def _make_f32_ones_producer(n_elems: int) -> TensorProducer:
    def _produce(f: BinaryIO) -> None:
        f.write(_F32_ONE * n_elems)

    return _produce


def _make_f32_rand_producer(n_elems: int, rng: random.Random) -> TensorProducer:
    def _produce(f: BinaryIO) -> None:
        vals = [rng.uniform(-1.0, 1.0) for _ in range(n_elems)]
        f.write(struct.pack(f"<{n_elems}f", *vals))

    return _produce


def generate_random_gguf(
    cfg: dict[str, object], quant: str, seed: int, out_path: str | Path
) -> None:
    """Validate config and write a deterministic random-weight GGUF file."""
    norm_cfg = validate_config(cfg, quant)
    quant_type = quant.upper()
    rng = random.Random(seed)

    metadata = _build_metadata(norm_cfg)
    tensor_items: list[tuple[str, tuple[int, ...], str, TensorProducer]] = []

    for name, shape, ttype in _tensor_specs(norm_cfg, quant_type):
        n_elems = math.prod(shape)
        if ttype in _QUANT_LAYOUT:
            producer = _make_quant_producer(n_elems, ttype, rng)
        elif name.endswith("ffn_gate_inp.weight"):
            producer = _make_f32_rand_producer(n_elems, rng)
        else:
            producer = _make_f32_ones_producer(n_elems)
        tensor_items.append((name, shape, ttype, producer))

    write_gguf(out_path, metadata, tensor_items, alignment=DEFAULT_ALIGNMENT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="models.random_gguf")
    parser.add_argument("--config", required=True, help="Path to model config JSON")
    parser.add_argument(
        "--quant", required=True, choices=["q8_0", "q4_0"], help="Quantization type"
    )
    parser.add_argument("--seed", required=True, type=int, help="PRNG seed")
    parser.add_argument("--out", required=True, help="Output GGUF file path")

    args = parser.parse_args(argv)
    try:
        raw_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        generate_random_gguf(raw_cfg, args.quant, args.seed, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
