"""Pruned draft vocabulary V' helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def load_vprime(
    ranking_json_path: str | Path,
    size: int = 32768,
    forced: Sequence[int] = (151643, 151645),
) -> list[int]:
    """Load top-`size` token IDs from a frequency ranking, ensuring `forced` IDs are included.

    For the `m` forced IDs absent from the top-`size` prefix, drops the `m` lowest-ranked
    non-forced IDs from the prefix (so forced IDs already present stay in place) and
    appends the missing forced IDs in the given order.
    """
    with open(ranking_json_path, "r", encoding="utf-8") as f:
        ranking = [int(tid) for tid in json.load(f)]

    if len(ranking) < size:
        raise ValueError(f"Ranking has {len(ranking)} tokens, fewer than size={size}")

    forced_list = list(dict.fromkeys(int(tid) for tid in forced))
    if len(forced_list) > size:
        raise ValueError(f"Number of unique forced IDs ({len(forced_list)}) exceeds size={size}")

    prefix = ranking[:size]
    prefix_set = set(prefix)
    forced_set = set(forced_list)

    missing = [tid for tid in forced_list if tid not in prefix_set]
    if not missing:
        return prefix

    drop_indices: set[int] = set()
    to_drop = len(missing)
    for idx in range(len(prefix) - 1, -1, -1):
        if prefix[idx] not in forced_set:
            drop_indices.add(idx)
            if len(drop_indices) == to_drop:
                break

    kept = [tid for idx, tid in enumerate(prefix) if idx not in drop_indices]
    return kept + missing


def vprime_map(vprime: Sequence[int], n_vocab: int) -> np.ndarray:
    """Return an int32 array of length `n_vocab` mapping target ID -> V' index (-1 outside)."""
    mapping = np.full(n_vocab, -1, dtype=np.int32)
    for idx, tid in enumerate(vprime):
        if tid < 0 or tid >= n_vocab:
            raise ValueError(f"Token ID {tid} out of range [0, {n_vocab})")
        mapping[tid] = idx
    return mapping
