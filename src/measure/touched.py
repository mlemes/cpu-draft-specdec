"""Touched-byte accounting and model sizing helpers (Python stdlib only)."""

from __future__ import annotations

from pathlib import Path

from models.gguf_io import read_gguf, tensor_nbytes

_QUANT_BPE: dict[str, float] = {
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}


def touched_bytes(path: str | Path) -> dict[str, int | float]:
    """Compute stored and per-token touched bytes for a GGUF model."""
    gguf = read_gguf(path)
    stored_bytes = sum(t.nbytes for t in gguf.tensors)

    token_embd = next((t for t in gguf.tensors if t.name == "token_embd.weight"), None)
    if token_embd is None:
        raise ValueError(f"Missing 'token_embd.weight' tensor in {path}")

    embedding_row_bytes = tensor_nbytes((token_embd.shape[0],), token_embd.ggml_type)
    expert_bytes = sum(t.nbytes for t in gguf.tensors if "_exps." in t.name)

    arch = gguf.metadata.get("general.architecture")
    if isinstance(arch, str) and arch:
        raw_count = gguf.metadata.get(f"{arch}.expert_count", 0)
        raw_used = gguf.metadata.get(f"{arch}.expert_used_count", 0)
        expert_count = int(raw_count) if isinstance(raw_count, int) else 0
        expert_used = int(raw_used) if isinstance(raw_used, int) else 0
    else:
        expert_count = 0
        expert_used = 0

    touched: int | float = stored_bytes - token_embd.nbytes + embedding_row_bytes
    if expert_count > 0:
        touched = touched - expert_bytes + (expert_bytes * expert_used) / expert_count

    return {
        "stored_bytes": stored_bytes,
        "embedding_row_bytes": embedding_row_bytes,
        "expert_bytes": expert_bytes,
        "expert_count": expert_count,
        "expert_used": expert_used,
        "touched_bytes_per_token": touched,
    }


def _predict_touched_bytes(d: int, bpe: float, vocab: int, layers: int) -> float:
    return (layers * (2.5 * d * d + 9.0 * d * d) + vocab * d) * bpe + (
        layers * 3 * d + d
    ) * 4.0


def config_for_touched_bytes(
    target_bytes: int | float,
    quant: str,
    vocab: int = 32768,
    layers: int = 12,
) -> dict[str, object]:
    """Return a qwen3 config for random_gguf whose predicted touched bytes are closest to target_bytes."""
    q_key = quant.lower()
    if q_key not in _QUANT_BPE:
        raise ValueError(f"Unsupported quant {quant!r}; expected 'q8_0' or 'q4_0'")
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be > 0, got {target_bytes}")
    if vocab < 259:
        raise ValueError(f"vocab must be >= 259, got {vocab}")
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")

    bpe = _QUANT_BPE[q_key]
    d = 256
    best_d = d
    best_err = abs(_predict_touched_bytes(d, bpe, vocab, layers) - target_bytes)

    while True:
        d += 128
        pred = _predict_touched_bytes(d, bpe, vocab, layers)
        err = abs(pred - target_bytes)
        if err < best_err:
            best_err = err
            best_d = d
        if pred > target_bytes and d > best_d:
            break

    head_dim = 64
    heads = best_d // head_dim
    max_kv = max(1, heads // 4)
    kv_heads = max(k for k in range(1, max_kv + 1) if heads % k == 0)

    return {
        "arch": "qwen3",
        "hidden": best_d,
        "layers": layers,
        "ffn": 3 * best_d,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "context": 4096,
    }
