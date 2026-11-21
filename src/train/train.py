"""Distillation trainer for dense Qwen3-layout Student (Stage B)."""

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
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from train.data import TeacherBatcher
from train.eval_alpha import DEFAULT_DOMAINS, evaluate
from train.losses import hard_ce, kd_forward_kl, tv_overlap
from train.student import Student, StudentConfig
from train.vprime import load_vprime, vprime_map


DEFAULT_VPRIME_RANKING = (
    "experiments/stage-a-feasibility/a5-vocab-coverage/"
    "results/a5-20261121T011928Z/ranking.json"
)


class UsageError(ValueError):
    """Raised for invalid CLI arguments or mismatched configuration (exit 2)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = proc.stdout.strip()
        return commit or None
    except Exception:
        return None


def _parse_data_spec(spec: str) -> list[tuple[str, float]]:
    if not spec or not spec.strip():
        raise UsageError("--data must not be empty")
    items: list[tuple[str, float]] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if ":" not in part:
            raise UsageError(f"Malformed --data entry (expected DIR:WEIGHT): {raw_part!r}")
        dir_str, weight_str = part.rsplit(":", 1)
        dir_str = dir_str.strip()
        if not dir_str:
            raise UsageError(f"Empty directory in --data entry: {raw_part!r}")
        try:
            weight = float(weight_str)
        except ValueError as exc:
            raise UsageError(f"Invalid weight in --data entry {raw_part!r}") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise UsageError(f"Weight must be finite and >= 0 in {raw_part!r}")
        items.append((dir_str, weight))
    if sum(w for _, w in items) <= 0.0:
        raise UsageError("--data weights must have positive sum")
    return items


def _parse_config_overrides(raw_json: str | None) -> dict[str, object]:
    if raw_json is None:
        return {}
    if not raw_json.lstrip().startswith("{"):
        # Not inline JSON: treat as a path to a JSON file.
        try:
            raw_json = Path(raw_json).read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"Cannot read --config file: {exc}") from exc
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise UsageError(f"Invalid --config JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise UsageError("--config must be a JSON object")
    valid_fields = {f.name for f in fields(StudentConfig)}
    unknown = set(parsed) - valid_fields
    if unknown:
        raise UsageError(f"Unknown StudentConfig fields in --config: {sorted(unknown)}")
    return parsed


def _lr_multiplier(step_1idx: int, total_steps: int, warmup: int, min_lr_frac: float) -> float:
    if warmup > 0 and step_1idx <= warmup:
        return step_1idx / float(warmup)
    decay_steps = max(1, total_steps - warmup)
    progress = min(1.0, (step_1idx - warmup) / float(decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_frac + (1.0 - min_lr_frac) * cosine


def _save_checkpoint(
    path: Path,
    cfg: StudentConfig,
    raw_model: Student,
    vprime: list[int],
    train_args: dict[str, object],
    step: int,
    tokens: int,
) -> None:
    state_dict = {
        k: v.detach().cpu().float() for k, v in raw_model.state_dict().items()
    }
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": state_dict,
            "vprime": list(vprime),
            "train_args": train_args,
            "step": int(step),
            "tokens": int(tokens),
        },
        path,
    )


def _write_manifest(
    path: Path,
    train_args: dict[str, object],
    git_commit: str | None,
    data_sha256: dict[str, str],
    start_time: str,
    end_time: str | None,
) -> None:
    payload = {
        "args": train_args,
        "git_commit": git_commit,
        "data_index_sha256": data_sha256,
        "start_time": start_time,
        "end_time": end_time,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distillation trainer for Student.")
    parser.add_argument(
        "--init",
        required=True,
        help="Checkpoint .pt path or 'scratch'",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON object of StudentConfig overrides",
    )
    parser.add_argument(
        "--vprime-ranking",
        default=DEFAULT_VPRIME_RANKING,
        help="Path to a5 ranking.json for scratch init",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Comma-separated DIR:WEIGHT list of teacher shard directories",
    )
    parser.add_argument("--out", required=True, help="Output run directory (must not exist)")
    parser.add_argument(
        "--loss",
        required=True,
        choices=("kd", "hard", "tv"),
        help="Training loss objective",
    )
    parser.add_argument("--tokens", type=int, default=1_000_000, help="Target token budget")
    parser.add_argument("--batch-seqs", type=int, default=16, help="Micro-batch sequences")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length override")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min-lr-frac", type=float, default=0.1, help="Final LR fraction")
    parser.add_argument("--warmup", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval-dir", default=None, help="Directory with eval_<domain>.jsonl")
    parser.add_argument("--eval-every-tokens", type=int, default=None, help="Eval interval")
    parser.add_argument("--ckpt-every-tokens", type=int, default=None, help="Checkpoint interval")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[list[tuple[str, float]], dict[str, object]]:
    out_dir = Path(args.out)
    if out_dir.exists():
        raise UsageError(f"Output directory already exists: {out_dir}")

    data_specs = _parse_data_spec(args.data)
    overrides = _parse_config_overrides(args.config)

    if args.tokens <= 0:
        raise UsageError(f"--tokens must be positive, got {args.tokens}")
    if args.batch_seqs <= 0:
        raise UsageError(f"--batch-seqs must be positive, got {args.batch_seqs}")
    if args.grad_accum <= 0:
        raise UsageError(f"--grad-accum must be positive, got {args.grad_accum}")
    if args.seq_len is not None and args.seq_len < 2:
        raise UsageError(f"--seq-len must be >= 2, got {args.seq_len}")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise UsageError(f"--lr must be finite and > 0, got {args.lr}")
    if not math.isfinite(args.min_lr_frac) or not (0.0 <= args.min_lr_frac <= 1.0):
        raise UsageError(f"--min-lr-frac must be in [0, 1], got {args.min_lr_frac}")
    if args.warmup < 0:
        raise UsageError(f"--warmup must be >= 0, got {args.warmup}")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise UsageError(f"--weight-decay must be finite and >= 0, got {args.weight_decay}")
    if args.eval_every_tokens is not None and args.eval_every_tokens <= 0:
        raise UsageError(f"--eval-every-tokens must be positive, got {args.eval_every_tokens}")
    if args.ckpt_every_tokens is not None and args.ckpt_every_tokens <= 0:
        raise UsageError(f"--ckpt-every-tokens must be positive, got {args.ckpt_every_tokens}")

    return data_specs, overrides


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        data_specs, overrides = _validate_args(args)

        device = torch.device(args.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if device.type == "cpu":
            torch.use_deterministic_algorithms(True)

        if args.init == "scratch":
            cfg = StudentConfig(**overrides)
            forced = tuple(f for f in (151643, 151645) if f < cfg.n_vocab_in)
            vprime = load_vprime(args.vprime_ranking, size=cfg.n_vprime, forced=forced)
            raw_model = Student(cfg)
        else:
            ckpt_path = Path(args.init)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = StudentConfig(**ckpt["config"])
            if overrides:
                merged_cfg = StudentConfig(**{**asdict(cfg), **overrides})
                if merged_cfg != cfg:
                    raise UsageError(
                        "--config overrides do not match checkpoint config"
                    )
            vprime = [int(x) for x in ckpt["vprime"]]
            if len(vprime) != cfg.n_vprime:
                raise UsageError(
                    f"Checkpoint vprime length ({len(vprime)}) != cfg.n_vprime ({cfg.n_vprime})"
                )
            raw_model = Student(cfg)
            raw_model.load_state_dict(ckpt["state_dict"])

        vmap_np = vprime_map(vprime, cfg.n_vocab_in)
        batcher = TeacherBatcher(
            data_specs, vmap_np, batch_seqs=args.batch_seqs, seed=args.seed
        )

        dom_seq_lens = [dom.seq_len for dom in batcher.domains]
        min_dom_seq_len = min(dom_seq_lens)
        if args.seq_len is None:
            if len(set(dom_seq_lens)) != 1:
                raise UsageError(
                    f"Data directories have differing seq_len {dom_seq_lens}; pass --seq-len"
                )
            seq_len = dom_seq_lens[0]
        else:
            if args.seq_len > min_dom_seq_len:
                raise UsageError(
                    f"--seq-len ({args.seq_len}) exceeds teacher data seq_len ({min_dom_seq_len})"
                )
            seq_len = args.seq_len

        raw_model.to(device)
        raw_model.train()
        model: torch.nn.Module = torch.compile(raw_model) if args.compile else raw_model

        vmap_t = (
            torch.from_numpy(vmap_np).to(device=device, dtype=torch.long)
            if args.loss == "hard"
            else None
        )

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "embed_tokens" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        tokens_per_step = args.batch_seqs * args.grad_accum * seq_len
        total_steps = max(1, math.ceil(args.tokens / tokens_per_step))
        # Warmup must leave decay steps: clamp to 10 % of the run when the requested warmup is too long (review 2026-11-21).
        warmup_steps = args.warmup if args.warmup < total_steps else max(1, total_steps // 10)
        if warmup_steps != args.warmup:
            print(f"warning: --warmup {args.warmup} >= {total_steps} steps; using {warmup_steps}", file=sys.stderr)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=False)

        train_args = {k: v for k, v in vars(args).items()}
        git_commit = _git_commit()
        data_sha256 = {
            d_str: _sha256_file(Path(d_str) / "index.json") for d_str, _ in data_specs
        }
        start_time = datetime.now(timezone.utc).isoformat()
        manifest_path = out_dir / "manifest.json"
        _write_manifest(
            manifest_path, train_args, git_commit, data_sha256, start_time, None
        )

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        metrics_path = out_dir / "metrics.jsonl"
        prev_tokens = 0
        last_eval_step = 0

        with open(metrics_path, "w", encoding="utf-8") as mf:
            for step in range(1, total_steps + 1):
                step_t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0

                for _ in range(args.grad_accum):
                    batch = next(batcher)
                    tokens = batch["tokens"]
                    topk_vidx = batch["topk_vidx"]
                    topk_p = batch["topk_p"]
                    loss_mask = batch["loss_mask"]

                    if seq_len < tokens.shape[1]:
                        tokens = tokens[:, :seq_len]
                        topk_vidx = topk_vidx[:, :seq_len, :]
                        topk_p = topk_p[:, :seq_len, :]
                        loss_mask = loss_mask[:, :seq_len].clone()
                        loss_mask[:, -1] = False

                    tokens = tokens.to(device=device, dtype=torch.long)
                    topk_vidx = topk_vidx.to(device=device, dtype=torch.long)
                    topk_p = topk_p.to(device=device, dtype=torch.float32)
                    loss_mask = loss_mask.to(device=device, dtype=torch.bool)

                    with autocast_ctx:
                        logits = model(tokens)
                        if args.loss == "kd":
                            loss = kd_forward_kl(logits, topk_vidx, topk_p, loss_mask)
                        elif args.loss == "hard":
                            assert vmap_t is not None
                            loss = hard_ce(logits, tokens, vmap_t, loss_mask)
                        else:
                            lower, _ = tv_overlap(logits, topk_vidx, topk_p)
                            mask_f = loss_mask.float()
                            mean_lower = (lower * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                            loss = 1.0 - mean_lower
                        scaled_loss = loss / float(args.grad_accum)

                    scaled_loss.backward()
                    loss_accum += float(loss.detach().item()) / float(args.grad_accum)

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0).item()
                )
                lr_mult = _lr_multiplier(step, total_steps, warmup_steps, args.min_lr_frac)
                current_lr = args.lr * lr_mult
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = max(1e-9, time.perf_counter() - step_t0)
                tokens_done = step * tokens_per_step
                tokens_per_s = tokens_per_step / dt

                step_rec = {
                    "step": step,
                    "tokens": tokens_done,
                    "loss": loss_accum,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "tokens_per_s": tokens_per_s,
                }
                mf.write(json.dumps(step_rec) + "\n")
                mf.flush()

                if (
                    args.ckpt_every_tokens is not None
                    and (tokens_done // args.ckpt_every_tokens)
                    > (prev_tokens // args.ckpt_every_tokens)
                ):
                    _save_checkpoint(
                        out_dir / f"ckpt-{tokens_done}.pt",
                        cfg,
                        raw_model,
                        vprime,
                        train_args,
                        step,
                        tokens_done,
                    )

                if (
                    args.eval_dir is not None
                    and args.eval_every_tokens is not None
                    and (tokens_done // args.eval_every_tokens)
                    > (prev_tokens // args.eval_every_tokens)
                ):
                    eval_summary = evaluate(
                        raw_model, vprime, args.eval_dir, DEFAULT_DOMAINS, device
                    )
                    mf.write(
                        json.dumps(
                            {"step": step, "tokens": tokens_done, "eval": eval_summary}
                        )
                        + "\n"
                    )
                    mf.flush()
                    last_eval_step = step

                prev_tokens = tokens_done

            final_tokens = total_steps * tokens_per_step
            if args.eval_dir is not None and last_eval_step != total_steps:
                eval_summary = evaluate(
                    raw_model, vprime, args.eval_dir, DEFAULT_DOMAINS, device
                )
                mf.write(
                    json.dumps(
                        {"step": total_steps, "tokens": final_tokens, "eval": eval_summary}
                    )
                    + "\n"
                )
                mf.flush()

        _save_checkpoint(
            out_dir / "final.pt",
            cfg,
            raw_model,
            vprime,
            train_args,
            total_steps,
            final_tokens,
        )
        end_time = datetime.now(timezone.utc).isoformat()
        _write_manifest(
            manifest_path, train_args, git_commit, data_sha256, start_time, end_time
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
