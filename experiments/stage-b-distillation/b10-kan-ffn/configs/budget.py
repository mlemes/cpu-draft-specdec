#!/usr/bin/env python3
"""b10: touched weight bytes per token for each FFN arm, at Q8_0 and Q4_0.

Pure arithmetic from the tensor layout (proposal section 4); no model is loaded. Only the FFN differs between arms;
attention, norms, the input-embedding row and the V' head are D-S. Usage: python budget.py [arms.json]
"""
import json
import sys
from pathlib import Path

BYTES_PER_WEIGHT = {"q8_0": 34 / 32, "q4_0": 18 / 32}  # ggml block of 32 weights: fp16 scale + 32 int8 / 16 bytes of nibbles
HIDDEN, N_LAYERS, Q_DIM, KV_DIM, HEAD_DIM, VPRIME = 512, 12, 512, 128, 128, 32768


def ffn_params(arm: dict) -> int:
    """Weights read per token in one layer's FFN."""
    h, t = arm["width"], arm["type"]
    if t == "swiglu":
        return 3 * HIDDEN * h
    if t in ("mlp", "mlp_rational", "gr_kan"):
        extra = arm.get("rational_groups", 0) * (arm.get("rational_m", 0) + 1 + arm.get("rational_n", 0))
        layers = 2 if t == "gr_kan" else 1  # GR-KAN puts a rational unit before each of its two linear maps
        return 2 * HIDDEN * h + layers * extra
    if t == "bspline_kan":
        per_edge = arm["grid"] + arm["order"] + 1  # spline coefficients plus the SiLU base weight
        return 2 * HIDDEN * h * per_edge
    raise ValueError(f"unknown ffn type {t}")


def other_params() -> int:
    attn = HIDDEN * Q_DIM + 2 * HIDDEN * KV_DIM + Q_DIM * HIDDEN
    norms = 2 * HIDDEN + 2 * HEAD_DIM  # pre-attention and pre-FFN RMSNorm, q/k norms
    return N_LAYERS * (attn + norms) + HIDDEN + HIDDEN * VPRIME + HIDDEN  # final norm, V' head, one embedding row


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("arms.json")
    arms = json.loads(path.read_text())["arms"]
    ref = None
    for arm in arms:
        p = N_LAYERS * ffn_params(arm) + other_params()  # norms counted as weights; they stay f32 in GGUF (small)
        ref = ref or p
        mb = {q: p * b / 1e6 for q, b in BYTES_PER_WEIGHT.items()}
        print(f"{arm['id']:<14} ffn/layer {ffn_params(arm):>9,}  total {p / 1e6:6.2f}M  "
              f"Q8_0 {mb['q8_0']:5.1f} MB  Q4_0 {mb['q4_0']:5.1f} MB  vs D-S {100 * (p - ref) / ref:+.2f}%")


if __name__ == "__main__":
    main()
