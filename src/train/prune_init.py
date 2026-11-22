"""Structured pruning initialization of a Student checkpoint from Qwen3-0.6B."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__":
    # Run as a script: drop src/train from sys.path so `train` resolves to the package, not this file.
    _here = str(_Path(__file__).resolve().parent)
    _sys.path[:] = [p for p in _sys.path if str(_Path(p or ".").resolve()) != _here]
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from train.student import StudentConfig
from train.vprime import load_vprime

# Head selection below assumes the default hidden size and 4/1 heads; only these fields may change.
PRUNE_OVERRIDES = {"n_layers", "ffn"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _topk_indices_stable(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Pick top-k indices with ties broken by smaller index, returned ascending."""
    order = np.argsort(-scores.detach().cpu().numpy().astype(np.float64), kind="mergesort")
    kept = np.sort(order[:k])
    return torch.from_numpy(kept.astype(np.int64))


def build_pruned_state_dict(
    src_weights: dict[str, torch.Tensor],
    vprime: list[int],
    cfg: StudentConfig,
) -> dict[str, torch.Tensor]:
    """Prune Qwen3-0.6B weights into a float32 Student state_dict."""
    embed = src_weights["model.embed_tokens.weight"].float()

    # Count source layers present in the checkpoint.
    n_src_layers = 0
    while f"model.layers.{n_src_layers}.input_layernorm.weight" in src_weights:
        n_src_layers += 1
    if n_src_layers == 0:
        raise ValueError("No model.layers.*.input_layernorm.weight keys found in source weights")

    # Hidden dimension importance: mean(|input_layernorm.weight|) * col L2 norm of embed_tokens.
    ln_abs_sum = torch.zeros(embed.shape[1], dtype=torch.float32)
    for l_idx in range(n_src_layers):
        ln_abs_sum += src_weights[f"model.layers.{l_idx}.input_layernorm.weight"].float().abs()
    mean_ln_abs = ln_abs_sum / float(n_src_layers)
    embed_col_norm = torch.linalg.vector_norm(embed, dim=0)
    hidden_importance = mean_ln_abs * embed_col_norm
    kept_hidden = _topk_indices_stable(hidden_importance, cfg.hidden)

    state_dict: dict[str, torch.Tensor] = {}
    embed_pruned = embed[:, kept_hidden].contiguous()
    state_dict["embed_tokens.weight"] = embed_pruned
    state_dict["norm.weight"] = src_weights["model.norm.weight"].float()[kept_hidden].contiguous()

    vprime_t = torch.tensor(vprime, dtype=torch.long)
    state_dict["lm_head.weight"] = embed_pruned[vprime_t].contiguous()

    src_n_kv_heads = 8
    head_dim = cfg.head_dim

    if cfg.n_layers < 2 or cfg.n_layers > n_src_layers:
        raise ValueError(f"n_layers must be in [2, {n_src_layers}], got {cfg.n_layers}")
    for i in range(cfg.n_layers):
        # Evenly spaced source layers, first and last always kept (12 layers: 0, 2, 5, ..., 27).
        src_l = round(i * (n_src_layers - 1) / (cfg.n_layers - 1))
        src_prefix = f"model.layers.{src_l}"
        dst_prefix = f"layers.{i}"

        state_dict[f"{dst_prefix}.input_layernorm.weight"] = (
            src_weights[f"{src_prefix}.input_layernorm.weight"].float()[kept_hidden].contiguous()
        )
        state_dict[f"{dst_prefix}.post_attention_layernorm.weight"] = (
            src_weights[f"{src_prefix}.post_attention_layernorm.weight"]
            .float()[kept_hidden]
            .contiguous()
        )

        # Attention projections & head selection.
        q_w = src_weights[f"{src_prefix}.self_attn.q_proj.weight"].float()
        k_w = src_weights[f"{src_prefix}.self_attn.k_proj.weight"].float()
        v_w = src_weights[f"{src_prefix}.self_attn.v_proj.weight"].float()
        o_w = src_weights[f"{src_prefix}.self_attn.o_proj.weight"].float()

        kv_scores = torch.empty(src_n_kv_heads, dtype=torch.float32)
        for kv in range(src_n_kv_heads):
            s0, s1 = kv * head_dim, (kv + 1) * head_dim
            kv_scores[kv] = torch.linalg.matrix_norm(k_w[s0:s1]) + torch.linalg.matrix_norm(
                v_w[s0:s1]
            )
        kv0 = int(torch.argmax(kv_scores).item())
        kv1 = (kv0 + 1) % src_n_kv_heads

        kept_q_heads = [2 * kv0, 2 * kv0 + 1, 2 * kv1, 2 * kv1 + 1]
        q_rows = torch.cat(
            [q_w[qh * head_dim : (qh + 1) * head_dim, kept_hidden] for qh in kept_q_heads], dim=0
        )
        o_cols = torch.cat(
            [o_w[kept_hidden, qh * head_dim : (qh + 1) * head_dim] for qh in kept_q_heads], dim=1
        )
        k_avg = 0.5 * (
            k_w[kv0 * head_dim : (kv0 + 1) * head_dim, kept_hidden]
            + k_w[kv1 * head_dim : (kv1 + 1) * head_dim, kept_hidden]
        )
        v_avg = 0.5 * (
            v_w[kv0 * head_dim : (kv0 + 1) * head_dim, kept_hidden]
            + v_w[kv1 * head_dim : (kv1 + 1) * head_dim, kept_hidden]
        )

        state_dict[f"{dst_prefix}.self_attn.q_proj.weight"] = q_rows.contiguous()
        state_dict[f"{dst_prefix}.self_attn.k_proj.weight"] = k_avg.contiguous()
        state_dict[f"{dst_prefix}.self_attn.v_proj.weight"] = v_avg.contiguous()
        state_dict[f"{dst_prefix}.self_attn.o_proj.weight"] = o_cols.contiguous()
        state_dict[f"{dst_prefix}.self_attn.q_norm.weight"] = (
            src_weights[f"{src_prefix}.self_attn.q_norm.weight"].float().clone()
        )
        state_dict[f"{dst_prefix}.self_attn.k_norm.weight"] = (
            src_weights[f"{src_prefix}.self_attn.k_norm.weight"].float().clone()
        )

        # FFN neuron selection.
        gate_w = src_weights[f"{src_prefix}.mlp.gate_proj.weight"].float()
        up_w = src_weights[f"{src_prefix}.mlp.up_proj.weight"].float()
        down_w = src_weights[f"{src_prefix}.mlp.down_proj.weight"].float()

        ffn_scores = torch.linalg.vector_norm(down_w, dim=0) * torch.linalg.vector_norm(
            up_w, dim=1
        )
        kept_ffn = _topk_indices_stable(ffn_scores, cfg.ffn)

        state_dict[f"{dst_prefix}.mlp.gate_proj.weight"] = (
            gate_w[kept_ffn][:, kept_hidden].contiguous()
        )
        state_dict[f"{dst_prefix}.mlp.up_proj.weight"] = (
            up_w[kept_ffn][:, kept_hidden].contiguous()
        )
        state_dict[f"{dst_prefix}.mlp.down_proj.weight"] = (
            down_w[kept_hidden][:, kept_ffn].contiguous()
        )

    return state_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pruned initialization of Student from Qwen3-0.6B.")
    parser.add_argument("--src", required=True, help="Source HF directory or model.safetensors path")
    parser.add_argument("--vprime-ranking", required=True, help="Path to a5 ranking.json")
    parser.add_argument("--out", required=True, help="Output checkpoint .pt path")
    parser.add_argument(
        "--config",
        default=None,
        help='JSON object of StudentConfig overrides; only n_layers and ffn, e.g. {"n_layers": 18}',
    )
    args = parser.parse_args(argv)

    src_path = Path(args.src)
    weights_path = src_path / "model.safetensors" if src_path.is_dir() else src_path

    overrides = json.loads(args.config) if args.config else {}
    if not isinstance(overrides, dict) or set(overrides) - PRUNE_OVERRIDES:
        parser.error(f"--config accepts only a JSON object with keys {sorted(PRUNE_OVERRIDES)}")
    cfg = StudentConfig(**overrides)
    vprime = load_vprime(args.vprime_ranking, size=cfg.n_vprime)
    source_sha256 = _sha256_file(weights_path)
    src_weights = load_file(str(weights_path), device="cpu")
    state_dict = build_pruned_state_dict(src_weights, vprime, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": state_dict,
            "vprime": list(vprime),
            "source_sha256": source_sha256,
        },
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
