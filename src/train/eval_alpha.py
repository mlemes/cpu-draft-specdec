"""Offline speculative decoding acceptance evaluation (Stage B)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__":
    # Run as a script: drop src/train from sys.path so `train` resolves to the package, not this file.
    _here = str(_Path(__file__).resolve().parent)
    _sys.path[:] = [p for p in _sys.path if str(_Path(p or ".").resolve()) != _here]
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import torch

from train.losses import tv_overlap
from train.student import Student, StudentConfig
from train.vprime import vprime_map


DEFAULT_DOMAINS = ("general", "pt_fin", "json")


def _normalize_domains(domains: str | Iterable[str]) -> list[str]:
    if isinstance(domains, str):
        parts = [d.strip() for d in domains.split(",")]
    else:
        parts = [str(d).strip() for d in domains]
    return [d for d in parts if d]


def _bootstrap_ci(
    prompt_means: list[float],
    boot_idx: np.ndarray,
) -> tuple[float, float]:
    arr = np.asarray(prompt_means, dtype=np.float64)
    boot_means = arr[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(
    model: torch.nn.Module,
    vprime: Sequence[int],
    eval_dir: str | Path,
    domains: str | Iterable[str],
    device: str | torch.device,
) -> dict[str, dict[str, object]]:
    """Evaluate student greedy agreement and TV overlap bounds per domain."""
    dev = torch.device(device)
    eval_path = Path(eval_dir)
    domain_list = _normalize_domains(domains)

    raw_model = getattr(model, "_orig_mod", model)
    cfg: StudentConfig = raw_model.cfg
    max_seq = int(cfg.max_seq)
    n_vocab_in = int(cfg.n_vocab_in)

    vprime_list = [int(x) for x in vprime]
    vprime_t = torch.tensor(vprime_list, dtype=torch.long, device=dev)
    vmap_np = vprime_map(vprime_list, n_vocab_in)

    was_training = model.training
    model.eval()
    results: dict[str, dict[str, object]] = {}

    try:
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if dev.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            for domain in domain_list:
                jsonl_path = eval_path / f"eval_{domain}.jsonl"
                if not jsonl_path.is_file():
                    continue

                prompt_agree_means: list[float] = []
                prompt_lower_means: list[float] = []
                prompt_upper_means: list[float] = []
                total_agree = 0.0
                total_lower = 0.0
                total_upper = 0.0
                total_positions = 0

                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        prompt_tokens = [int(t) for t in row["prompt_tokens"]]
                        gen_tokens = [int(t) for t in row["gen_tokens"]]
                        n_gen = len(gen_tokens)
                        if n_gen == 0:
                            continue

                        full_input = prompt_tokens + gen_tokens[:-1]
                        if not full_input:
                            continue

                        input_ids = full_input[-max_seq:]
                        kept = min(n_gen, len(input_ids))

                        tokens_t = torch.tensor(
                            input_ids, dtype=torch.long, device=dev
                        ).unsqueeze(0)
                        with autocast_ctx:
                            logits = model(tokens_t)[0, -kept:, :].float()

                        top_ids_np = np.asarray(row["top_ids"][-kept:], dtype=np.int64)
                        top_logp_np = np.asarray(row["top_logp"][-kept:], dtype=np.float32)

                        in_range = (top_ids_np >= 0) & (top_ids_np < len(vmap_np))
                        safe_ids = np.where(in_range, top_ids_np, 0)
                        topk_vidx_np = np.where(in_range, vmap_np[safe_ids], -1).astype(
                            np.int64, copy=False
                        )

                        topk_vidx_t = torch.from_numpy(topk_vidx_np).to(
                            device=dev, dtype=torch.long
                        )
                        topk_p_t = torch.from_numpy(np.exp(top_logp_np)).to(
                            device=dev, dtype=torch.float32
                        )

                        lower_t, upper_t = tv_overlap(logits, topk_vidx_t, topk_p_t)

                        pred_ids_t = vprime_t[logits.argmax(dim=-1)]
                        target_top0_t = torch.from_numpy(top_ids_np[:, 0]).to(
                            device=dev, dtype=torch.long
                        )
                        agree_t = (pred_ids_t == target_top0_t).float()

                        prompt_agree_means.append(float(agree_t.mean().item()))
                        prompt_lower_means.append(float(lower_t.mean().item()))
                        prompt_upper_means.append(float(upper_t.mean().item()))

                        total_agree += float(agree_t.sum().item())
                        total_lower += float(lower_t.sum().item())
                        total_upper += float(upper_t.sum().item())
                        total_positions += kept

                n_prompts = len(prompt_agree_means)
                if n_prompts == 0 or total_positions == 0:
                    results[domain] = {
                        "positions": 0,
                        "prompts": 0,
                        "greedy_agree": [0.0, 0.0, 0.0],
                        "tv_lower": [0.0, 0.0, 0.0],
                        "tv_upper": [0.0, 0.0, 0.0],
                    }
                    continue

                rng = np.random.default_rng(0)
                boot_idx = rng.integers(0, n_prompts, size=(2000, n_prompts))

                agree_mean = total_agree / total_positions
                lower_mean = total_lower / total_positions
                upper_mean = total_upper / total_positions

                agree_lo, agree_hi = _bootstrap_ci(prompt_agree_means, boot_idx)
                lower_lo, lower_hi = _bootstrap_ci(prompt_lower_means, boot_idx)
                upper_lo, upper_hi = _bootstrap_ci(prompt_upper_means, boot_idx)

                results[domain] = {
                    "positions": int(total_positions),
                    "prompts": int(n_prompts),
                    "greedy_agree": [float(agree_mean), agree_lo, agree_hi],
                    "tv_lower": [float(lower_mean), lower_lo, lower_hi],
                    "tv_upper": [float(upper_mean), upper_lo, upper_hi],
                }
    finally:
        model.train(was_training)

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate student offline acceptance against target top-k logs."
    )
    parser.add_argument("--ckpt", required=True, help="Path to student checkpoint .pt")
    parser.add_argument("--eval-dir", required=True, help="Directory with eval_<domain>.jsonl")
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated domain names (default: general,pt_fin,json)",
    )
    parser.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size option accepted for CLI compatibility (default: 16)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.batch <= 0:
        print(f"error: --batch must be positive, got {args.batch}", file=sys.stderr)
        return 2

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = StudentConfig(**ckpt["config"])
    model = Student(cfg)
    model.load_state_dict(ckpt["state_dict"])
    dev = torch.device(args.device)
    model.to(dev)

    summary = evaluate(
        model=model,
        vprime=ckpt["vprime"],
        eval_dir=args.eval_dir,
        domains=args.domains,
        device=dev,
    )
    payload = json.dumps(summary, indent=2)

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
