"""Memmapped teacher top-k shard reader and multi-domain batcher."""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import torch


class TeacherShards:
    """Read-only memmap view over a teacher_score output directory."""

    def __init__(self, shard_dir: str | Path) -> None:
        self.dir = Path(shard_dir)
        with open(self.dir / "index.json", "r", encoding="utf-8") as f:
            self.index = json.load(f)

        self.seq_len: int = int(self.index["seq_len"])
        self.k: int = int(self.index["k"])
        self.n_vocab: int = int(self.index["n_vocab"])

        self._tokens: list[np.memmap] = []
        self._ids: list[np.memmap] = []
        self._logp: list[np.memmap] = []
        self._cum_seqs: list[int] = []

        total = 0
        for shard in self.index["shards"]:
            name = shard["name"]
            n_seq = int(shard["n_seq"])
            self._tokens.append(
                np.memmap(
                    self.dir / f"{name}.tokens.u32",
                    dtype="<u4",
                    mode="r",
                    shape=(n_seq, self.seq_len),
                )
            )
            self._ids.append(
                np.memmap(
                    self.dir / f"{name}.ids.u32",
                    dtype="<u4",
                    mode="r",
                    shape=(n_seq, self.seq_len, self.k),
                )
            )
            self._logp.append(
                np.memmap(
                    self.dir / f"{name}.logp.f16",
                    dtype="<f2",
                    mode="r",
                    shape=(n_seq, self.seq_len, self.k),
                )
            )
            total += n_seq
            self._cum_seqs.append(total)

        self.n_seq = total

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx < 0 or idx >= self.n_seq:
            raise IndexError(f"Sequence index {idx} out of range [0, {self.n_seq})")
        shard_idx = bisect_right(self._cum_seqs, idx)
        base = 0 if shard_idx == 0 else self._cum_seqs[shard_idx - 1]
        local_idx = idx - base
        return (
            self._tokens[shard_idx][local_idx],
            self._ids[shard_idx][local_idx],
            self._logp[shard_idx][local_idx],
        )


class TeacherBatcher:
    """Infinite deterministic multi-domain batcher over TeacherShards."""

    def __init__(
        self,
        dirs_with_weights: list[tuple[str, float]],
        vprime_map: np.ndarray,
        batch_seqs: int,
        seed: int,
    ) -> None:
        if not dirs_with_weights:
            raise ValueError("dirs_with_weights must not be empty")
        if batch_seqs <= 0:
            raise ValueError(f"batch_seqs must be positive, got {batch_seqs}")

        self.domains = [TeacherShards(d) for d, _ in dirs_with_weights]
        weights = np.asarray([w for _, w in dirs_with_weights], dtype=np.float64)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Domain weights must be non-negative with positive sum")
        self.probs = weights / weights.sum()

        for dom in self.domains:
            if len(dom) == 0:
                raise ValueError(f"Domain {dom.dir} contains 0 sequences")

        seq_lens = {dom.seq_len for dom in self.domains}
        if len(seq_lens) != 1:
            raise ValueError(
                f"Teacher directories must share one seq_len, got {[dom.seq_len for dom in self.domains]}"
            )

        self.vprime_map = np.asarray(vprime_map, dtype=np.int32)
        self.batch_seqs = batch_seqs
        self.rng = np.random.default_rng(seed)

        self._perms = [self.rng.permutation(len(dom)) for dom in self.domains]
        self._cursors = [0 for _ in self.domains]

    def _next_seq_idx(self, dom_idx: int) -> int:
        cursor = self._cursors[dom_idx]
        perm = self._perms[dom_idx]
        if cursor >= len(perm):
            perm = self.rng.permutation(len(self.domains[dom_idx]))
            self._perms[dom_idx] = perm
            cursor = 0
        seq_idx = int(perm[cursor])
        self._cursors[dom_idx] = cursor + 1
        return seq_idx

    def __iter__(self) -> TeacherBatcher:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        dom_choices = self.rng.choice(len(self.domains), size=self.batch_seqs, p=self.probs)

        tokens_list: list[np.ndarray] = []
        ids_list: list[np.ndarray] = []
        logp_list: list[np.ndarray] = []

        for dom_idx in dom_choices:
            seq_idx = self._next_seq_idx(int(dom_idx))
            tok, ids, logp = self.domains[int(dom_idx)][seq_idx]
            tokens_list.append(tok)
            ids_list.append(ids)
            logp_list.append(logp)

        tokens_np = np.stack(tokens_list, axis=0).astype(np.int64, copy=False)
        ids_np = np.stack(ids_list, axis=0)
        logp_np = np.stack(logp_list, axis=0).astype(np.float32, copy=False)

        n_vocab = len(self.vprime_map)
        in_range = ids_np < n_vocab
        safe_ids = np.where(in_range, ids_np, 0)
        mapped = np.where(in_range, self.vprime_map[safe_ids], -1).astype(np.int64, copy=False)

        topk_p = np.exp(logp_np)
        loss_mask = torch.ones(tokens_np.shape, dtype=torch.bool)
        loss_mask[:, -1] = False

        return {
            "tokens": torch.from_numpy(tokens_np),
            "topk_vidx": torch.from_numpy(mapped),
            "topk_p": torch.from_numpy(topk_p),
            "loss_mask": loss_mask,
        }
