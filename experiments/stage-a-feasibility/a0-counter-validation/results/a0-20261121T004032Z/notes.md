# a0 run 2026-11-21T00:40:32Z: findings

Setup: SMT off, governor `performance`, stream_bench pinned to CPU 2, 5 timed passes per run, root group restricted to the top 2 L3 ways (12 MB) during the run, benchmark group swept over 1, 2, 4, 8 and 14 ways (6–84 MB), buffers 4–384 MiB. perf attached per process with `ls_dmnd_fills_from_sys.{int_cache,mem_io_local}` and `ls_any_fills_from_sys.{int_cache,mem_io_local}`. Run time 15 s. Root L3 mask restored to `ffff` afterwards.

`summary.json` reports `passed: false`. The criterion in the spec assumed an LRU-like cliff (full miss once the buffer is twice the allocation) and a near-zero floor; neither holds on this part. The counter findings:

| Finding | Evidence | Label |
|---|---|---|
| CAT limits L3 capacity as configured | `llc_occupancy` saturates at 6.0, 12.0, 24.0, 47.9 and 83.5 MB for 1, 2, 4, 8 and 14 ways | [V] |
| MBM counts DRAM read bytes accurately | 1 way: MBM bytes per pass = 1.00–1.01 × buffer for every buffer ≥ 16 MiB | [V] |
| No cache cliff | Buffers at 2× the allocation still hit L3 for 10–30% of their bytes; the miss fraction rises gradually (for example 14 ways: 0.62 at 128 MiB, 0.79 at 192 MiB, 0.87 at 384 MiB). Consistent with a non-LRU replacement policy | [V] data, [U] cause |
| Floor for resident buffers | Buffers that fit in the allocation still show about 0.21–0.23 × buffer bytes of MBM traffic per pass, independent of the allocation | [U] |
| Core fill-source events are not DRAM evidence | `ls_dmnd_fills_from_sys.mem_io_local` ≈ 0 in every run; `ls_any_fills_from_sys.mem_io_local` ≈ 0.8 × MBM bytes; `int_cache` ≈ 0. The L2 hardware prefetcher fills most lines before demand, so L1 fill-source events miss them | [V] |
| MBM warm-up | A new resctrl group reads `Unavailable` for about its first 0.5 s of activity; the driver now warms the group up and retries reads | [V] |

Implications for the project:

1. MBM bytes per token and `llc_occupancy` are the primary cache evidence. Core fill-source events are supplementary only.
2. Residency claims must be stated as DRAM bytes per token versus allocation curves. A sharp knee should not be expected; the a2 and b5 analyses need a soft-transition model, not a threshold.
3. The 0.22 floor must be explained (L3 uncore `l3_misses` and `l3_cache_accesses` in a2) before small residual DRAM traffic is attributed to the model.

Decision on the a0 gate is pending with the operator: either revise the pass criterion to MBM accuracy at full miss plus monotone, allocation-dependent traffic, or investigate further first.
