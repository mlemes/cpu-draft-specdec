#!/usr/bin/env python3
"""b10 analysis: per-prompt offline greedy agreement for each arm, paired bootstrap differences.

Every model is scored on the same eval prompts (the a4 prompt sets, target greedy continuations), so arm
differences are paired by prompt. A difference's 95% CI resamples prompts (2000 draws, seed 0) and weights
each prompt by its position count, as in src/train/eval_alpha.py.

Usage: PYTHONPATH=src python analyze.py --eval-dir DIR --out OUT.json NAME=CKPT [NAME=CKPT ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from train.student import Student, StudentConfig

DOMAINS = ("general", "pt_fin", "json")
PAIRS = [  # (candidate, reference): the comparisons the b10 design names
    ("ds-fresh", "ds-cont"), ("mlp-silu", "ds-fresh"), ("mlp-rational", "mlp-silu"), ("gr-kan", "mlp-rational"),
    ("kan-bspline", "mlp-silu"), ("kan-bspline", "ds-fresh"), ("gr-kan", "ds-fresh"), ("gr-kan", "mlp-silu"),
    ("kan-bspline-fit", "kan-bspline"), ("kan-bspline-fit", "ds-cont"), ("kan-bspline-fit", "ds-fresh"),
    ("kan-bspline-fit", "mlp-silu"), ("ds-cont", "ds-ref"),
]


def per_prompt(model: Student, vprime: list[int], path: Path, device: torch.device, limit: int | None):
    """Per-prompt agreement: integer agree counts and position counts, as in eval_alpha.evaluate()."""
    vp_t = torch.tensor(vprime, dtype=torch.long, device=device)
    agree, count = [], []
    seen = 0
    with path.open(encoding="utf-8") as f, torch.no_grad():
        for line in f:
            line = line.strip()
            if not line:  # skip blank lines, as eval_alpha does
                continue
            if limit is not None and seen >= limit:
                break
            seen += 1
            row = json.loads(line)
            prompt, gen = [int(t) for t in row["prompt_tokens"]], [int(t) for t in row["gen_tokens"]]
            inp = (prompt + gen[:-1])[-model.cfg.max_seq:]
            kept = min(len(gen), len(inp))
            if kept == 0:
                continue
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
            with ctx:
                logits = model(torch.tensor(inp, device=device)[None])[0, -kept:].float()
            top0 = torch.tensor([r[0] for r in row["top_ids"][-kept:]], device=device)
            a = (vp_t[logits.argmax(-1)] == top0).float()
            agree.append(int(a.sum().item()))
            count.append(int(a.numel()))
    return np.asarray(agree, dtype=np.int64), np.asarray(count, dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="first N prompts per domain (smoke tests)")
    p.add_argument("models", nargs="+", help="NAME=CKPT")
    a = p.parse_args()
    dev = torch.device(a.device)
    # scores[name][domain] = (per-prompt agree counts, per-prompt position counts)
    scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for spec in a.models:
        name, ckpt_path = spec.split("=", 1)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = Student(StudentConfig(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        m.to(dev).eval()
        vp = [int(t) for t in ck["vprime"]]
        scores[name] = {d: per_prompt(m, vp, Path(a.eval_dir) / f"eval_{d}.jsonl", dev, a.limit) for d in DOMAINS}
        del m, ck
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(name, {d: round(float(s[0].sum() / s[1].sum()), 4) for d, s in scores[name].items()}, file=sys.stderr)

    out: dict[str, object] = {"arms": {}, "pairs": []}
    for d in DOMAINS:
        # One resample index per domain, drawn exactly as eval_alpha.evaluate() does, shared by all arms and pairs.
        n = len(next(iter(scores.values()))[d][0])
        if n == 0:
            continue
        rng = np.random.default_rng(0)
        boot_idx = rng.integers(0, n, size=(2000, n))
        for name, s in scores.items():
            cnt, w = s[d]
            # Point estimate and CI are position-weighted: total agree count / total positions.
            boot = cnt[boot_idx].sum(1) / w[boot_idx].sum(1)
            out["arms"].setdefault(name, {})[d] = [float(cnt.sum() / w.sum()), *map(float, np.percentile(boot, [2.5, 97.5]))]
        for cand, ref in PAIRS:
            if cand not in scores or ref not in scores:
                continue
            (cc, w), (cr, wr) = scores[cand][d], scores[ref][d]
            assert np.array_equal(w, wr), "prompt sets differ"
            diff = (cc - cr).astype(np.float64)  # per-prompt agree-count difference, weighted by w below
            boot = diff[boot_idx].sum(1) / w[boot_idx].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out["pairs"].append({"domain": d, "candidate": cand, "reference": ref, "delta": float(diff.sum() / w.sum()),
                                 "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)})
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
