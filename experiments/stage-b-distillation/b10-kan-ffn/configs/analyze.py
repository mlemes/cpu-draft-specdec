#!/usr/bin/env python3
"""b10 analysis: per-prompt offline greedy agreement for each arm, paired bootstrap differences.

Every model is scored on the same eval prompts (the a4 prompt sets, target greedy continuations), so arm
differences are paired by prompt. A difference's 95% CI resamples prompts (2000 draws, seed 0) and weights
each prompt by its position count, as in src/train/eval_alpha.py.

Multi-seed arms are named NAME@s<k> (k >= 0, e.g. ds-fresh@s1); bare NAME is seed 0. summarize() reports each
seed's paired difference and pools seeds that share an arm.

Usage: PYTHONPATH=src python analyze.py --eval-dir DIR --out OUT.json [--cache-dir DIR] NAME=CKPT [NAME=CKPT ...]

NAME is an arm name with an optional seed suffix NAME@s<k> (bare NAME means seed 0). --cache-dir DIR reuses
per-model scores, keyed by a hash of the checkpoint, the eval files, --limit and the device type.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DOMAINS = ("general", "pt_fin", "json")
PAIRS = [  # (candidate, reference): the comparisons the b10 design names
    ("ds-fresh", "ds-cont"), ("mlp-silu", "ds-fresh"), ("mlp-rational", "mlp-silu"), ("gr-kan", "mlp-rational"),
    ("kan-bspline", "mlp-silu"), ("kan-bspline", "ds-fresh"), ("gr-kan", "ds-fresh"), ("gr-kan", "mlp-silu"),
    ("kan-bspline-fit", "kan-bspline"), ("kan-bspline-fit", "ds-cont"), ("kan-bspline-fit", "ds-fresh"),
    ("kan-bspline-fit", "mlp-silu"), ("ds-cont", "ds-ref"),
    ("ds-fresh", "ds-ref"), ("gr-kan-n1", "mlp-silu-n1"), ("mlp-silu-n1", "mlp-silu"), ("gr-kan-n1", "mlp-silu"),
    ("gr-kan-n1", "ds-fresh"),
    ("mlp-rational-n1", "mlp-silu-n1"), ("gr-kan-n1", "mlp-rational-n1"), ("mlp-rational-n1", "ds-fresh"),
]
CACHE_CHUNK = 16 * 1024 * 1024


def parse_name(name: str) -> tuple[str, int]:
    """Split a run name into (arm, seed): 'arm' -> ('arm', 0), 'arm@s<k>' -> ('arm', k)."""
    if "@" not in name:
        return name, 0
    arm, _, suffix = name.rpartition("@")
    digits = suffix[1:] if suffix.startswith("s") else ""
    if not digits or any(c not in "0123456789" for c in digits):
        raise ValueError(f"bad run name {name!r}: expected NAME or NAME@s<k> with k a non-negative integer")
    return arm, int(digits)


def per_prompt(model, vprime: list[int], path: Path, device, limit: int | None):
    """Per-prompt agreement: integer agree counts and position counts, as in eval_alpha.evaluate()."""
    import torch

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


def summarize(scores, pairs=PAIRS) -> dict:
    """Per-domain arm scores, per-seed paired differences and cross-seed pooling (pure numpy, no torch).

    scores = {name: {domain: (per-prompt agree counts, per-prompt position counts)}}. One bootstrap resample
    index is drawn per domain (2000 draws, seed 0) and shared by every arm and pair, as in eval_alpha.
    """
    # arm -> {seed: name}. The bare name and 'arm@s0' both mean seed 0: the bare name wins.
    groups: dict[str, dict[int, str]] = {}
    for name in scores:
        arm, seed = parse_name(name)
        slots = groups.setdefault(arm, {})
        prev = slots.get(seed)
        if prev is not None:
            if seed == 0 and ("@" not in name or "@" not in prev):
                # Both 'arm' and 'arm@s0' mean seed 0; the bare name is canonical, so keep whichever is bare.
                if "@" not in name:
                    slots[seed] = name
                continue
            raise ValueError(f"duplicate seed {seed} for arm {arm!r}: {prev!r} and {name!r}")
        slots[seed] = name

    out: dict[str, object] = {"arms": {}, "pairs": [], "pooled": []}
    weights: dict[str, np.ndarray] = {}
    boots: dict[str, np.ndarray] = {}
    pooled_diffs: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    for d in DOMAINS:
        names = [nm for nm in scores if d in scores[nm]]
        if not names:
            continue
        # One resample index per domain, drawn exactly as eval_alpha.evaluate() does, shared by all arms and pairs.
        n = len(scores[names[0]][d][0])
        if n == 0:
            continue
        w = scores[names[0]][d][1]
        for nm in names[1:]:
            assert np.array_equal(w, scores[nm][d][1]), f"prompt sets differ for {nm!r} in {d!r}"
        weights[d] = w
        rng = np.random.default_rng(0)
        boot_idx = rng.integers(0, n, size=(2000, n))
        boots[d] = boot_idx
        for nm in names:
            cnt, w_nm = scores[nm][d]
            # Point estimate and CI are position-weighted: total agree count / total positions.
            boot = cnt[boot_idx].sum(1) / w_nm[boot_idx].sum(1)
            out["arms"].setdefault(nm, {})[d] = [float(cnt.sum() / w_nm.sum()), *map(float, np.percentile(boot, [2.5, 97.5]))]
        for cand, ref in pairs:
            cand_seeds, ref_seeds = groups.get(cand, {}), groups.get(ref, {})
            for seed in sorted(set(cand_seeds) & set(ref_seeds)):
                cand_name, ref_name = cand_seeds[seed], ref_seeds[seed]
                if d not in scores[cand_name] or d not in scores[ref_name]:
                    continue
                (cc, cw), (cr, _) = scores[cand_name][d], scores[ref_name][d]
                # Per-prompt agree-count difference, weighted by the position counts below.
                diff = (cc - cr).astype(np.float64)
                boot = diff[boot_idx].sum(1) / cw[boot_idx].sum(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                out["pairs"].append({"domain": d, "candidate": cand, "reference": ref,
                                     "delta": float(diff.sum() / cw.sum()), "ci95": [float(lo), float(hi)],
                                     "excludes_zero": bool(lo > 0 or hi < 0), "seed": seed})
                pooled_diffs.setdefault((cand, ref, d), {})[seed] = diff

    for (cand, ref, d), per_seed in pooled_diffs.items():
        seeds = sorted(per_seed)
        if len(seeds) < 2:
            continue
        w, boot_idx = weights[d], boots[d]
        diff = np.sum([per_seed[s] for s in seeds], axis=0, dtype=np.float64)
        boot = diff[boot_idx].sum(1) / (len(seeds) * w[boot_idx].sum(1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        seed_deltas = [float(per_seed[s].sum() / w.sum()) for s in seeds]
        out["pooled"].append({"domain": d, "candidate": cand, "reference": ref,
                              "delta": float(diff.sum() / (len(seeds) * w.sum())),
                              "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0),
                              "seeds": seeds, "seed_deltas": seed_deltas,
                              "seed_sd": float(np.std(seed_deltas, ddof=1))})
    return out


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in 16 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CACHE_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def score_key(ckpt: Path, eval_dir: Path, limit: int | None, device_type: str) -> str:
    """Cache key over checkpoint bytes, eval files, --limit and the device type."""
    h = hashlib.sha256()
    h.update(file_sha256(ckpt).encode())
    for d in DOMAINS:
        h.update(file_sha256(eval_dir / f"eval_{d}.jsonl").encode())
    h.update(str(limit).encode())
    h.update(device_type.encode())
    return h.hexdigest()


def main() -> int:
    import torch

    from train.student import Student, StudentConfig

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="first N prompts per domain (smoke tests)")
    p.add_argument("--cache-dir", default=None, help="reuse per-model scores keyed by a content hash")
    p.add_argument("models", nargs="+", help="NAME=CKPT, NAME optionally suffixed @s<k>")
    a = p.parse_args()
    dev = torch.device(a.device)
    eval_dir = Path(a.eval_dir)
    # scores[name][domain] = (per-prompt agree counts, per-prompt position counts)
    scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for spec in a.models:
        name, ckpt_path = spec.split("=", 1)
        ckpt = Path(ckpt_path)
        cache_path = None
        if a.cache_dir:
            key = score_key(ckpt, eval_dir, a.limit, dev.type)
            cache_path = Path(a.cache_dir) / f"{key}.npz"
            if cache_path.exists():
                with np.load(cache_path) as z:
                    scores[name] = {d: (z[f"{d}_agree"].astype(np.int64), z[f"{d}_count"].astype(np.int64))
                                    for d in DOMAINS}
                print(f"{name}: cached {cache_path}", file=sys.stderr)
                print(name, {d: round(float(s[0].sum() / s[1].sum()), 4) for d, s in scores[name].items()},
                      file=sys.stderr)
                continue
            print(f"{name}: not cached, scoring", file=sys.stderr)
        else:
            print(f"{name}: no --cache-dir, scoring", file=sys.stderr)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = Student(StudentConfig(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        m.to(dev).eval()
        vp = [int(t) for t in ck["vprime"]]
        scores[name] = {d: per_prompt(m, vp, eval_dir / f"eval_{d}.jsonl", dev, a.limit) for d in DOMAINS}
        del m, ck
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(name, {d: round(float(s[0].sum() / s[1].sum()), 4) for d, s in scores[name].items()}, file=sys.stderr)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file, then rename: an interrupted write never leaves a truncated cache entry.
            tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez(tmp, **{f"{d}_{k}": scores[name][d][i] for d in DOMAINS
                             for i, k in ((0, "agree"), (1, "count"))})
            tmp.replace(cache_path)

    Path(a.out).write_text(json.dumps(summarize(scores), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
