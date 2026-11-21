"""Bootstrap confidence intervals and speculative-decoding analytical helpers."""

from __future__ import annotations

import random
from typing import Iterable


def _quantile(sorted_vals: list[float], q: float) -> float:
    m = len(sorted_vals)
    pos = q * (m - 1)
    idx = int(pos)
    frac = pos - idx
    if idx + 1 < m:
        return sorted_vals[idx] + frac * (sorted_vals[idx + 1] - sorted_vals[idx])
    return sorted_vals[idx]


def bootstrap_ci(
    x: Iterable[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    vals = [float(v) for v in x]
    if not vals:
        raise ValueError("input sequence must not be empty")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    n = len(vals)
    point = sum(vals) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    lo = _quantile(means, alpha / 2.0)
    hi = _quantile(means, 1.0 - alpha / 2.0)
    return point, lo, hi


def paired_bootstrap_ratio_ci(
    a: Iterable[float],
    b: Iterable[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    a_vals = [float(v) for v in a]
    b_vals = [float(v) for v in b]
    if not a_vals or len(a_vals) != len(b_vals):
        raise ValueError("inputs must be non-empty and have equal length")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    n = len(a_vals)
    sum_b = sum(b_vals)
    if sum_b <= 0.0:
        raise ValueError("denominator mean must be positive")
    ratio = sum(a_vals) / sum_b

    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        boot_a = sum(a_vals[i] for i in indices)
        boot_b = sum(b_vals[i] for i in indices)
        if boot_b <= 0.0:
            raise ValueError("bootstrap replicate has non-positive denominator mean")
        ratios.append(boot_a / boot_b)

    ratios.sort()
    lo = _quantile(ratios, alpha / 2.0)
    hi = _quantile(ratios, 1.0 - alpha / 2.0)
    return ratio, lo, hi


def expected_tokens(alpha: float, gamma: float | int) -> float:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(
    alpha: float,
    gamma: float | int,
    t_d: float,
    t_v: float,
    t_t: float,
    t_overhead: float = 0.0,
) -> float:
    denom = gamma * t_d + t_v + t_overhead
    if denom <= 0.0:
        raise ValueError("denominator gamma*t_d + t_v + t_overhead must be positive")
    return expected_tokens(alpha, gamma) * t_t / denom


def break_even_alpha(
    gamma: float | int,
    t_d: float,
    t_v: float,
    t_t: float,
    t_overhead: float = 0.0,
    target: float = 1.0,
    tol: float = 1e-4,
) -> float | None:
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if speedup(1.0, gamma, t_d, t_v, t_t, t_overhead) < target:
        return None
    if speedup(0.0, gamma, t_d, t_v, t_t, t_overhead) >= target:
        return 0.0

    lo = 0.0
    hi = 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if speedup(mid, gamma, t_d, t_v, t_t, t_overhead) >= target:
            hi = mid
        else:
            lo = mid
    return hi
