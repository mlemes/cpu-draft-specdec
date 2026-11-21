# a2 run 2026-11-21T01:09:36Z: findings

Random-weight qwen3 models in the D-S layout (12 layers, vocab 32768 output head), widths chosen to hit target touched bytes. llama-bench CPU-only (`-dev none`), SMT off, threads pinned to CPUs 0..T−1, benchmark group `a2bench`, root group on the top 2 L3 ways. Grid phase at the full 84 MB allocation (14 ways); cat phase at 6 threads with 1–14 ways. 78 configurations, no errors; root L3 mask restored.

| Finding | Evidence | Label |
|---|---|---|
| Models touching ≤ 56 MB per token are L3-resident | MBM ratio (DRAM bytes per token / touched bytes) 0.00–0.05 at 18–60 MB touched, 84 MB allocation | [V] |
| DRAM traffic grows smoothly past L3 | MBM ratio 0.25–0.33 at 100–113 MB, 0.38–0.49 at 123–147 MB, 0.77 at 283–285 MB, 0.89–0.91 at 533–556 MB | [V] |
| Cache capacity alone changes latency | 56 MB Q8_0, 6 threads: 0.95 ms/token at 14 ways (ratio 0.04) versus 2.41 ms at 1 way (ratio 1.05), same compute: 2.5× | [V] |
| Draft time at the D-S size | 56 MB Q8_0: 1.14 ms/token at 3 threads, 0.97 ms at 6 threads; 43 MB Q4_0: 1.20–1.68 ms | [V] |
| 6 threads are unstable | Relative standard deviation up to 50% at 6 threads versus mostly < 10% at 3 threads; all 6 cores are shared with the OS and the root group | [V] |
| Q4_0 is not faster than Q8_0 when resident | 16–64 MB: Q4_0 equal or slower than Q8_0 per token (dequantization cost, no VNNI); Q4_0 helps only when DRAM-bound by bytes | [V] |
| DRAM-bound rate | 533 MB Q8_0 at 3 threads: 11.5 ms/token, about 41 GB/s | [V] |
| The a0 floor (0.22) does not appear for model weights | resident models show 0.00–0.05; the floor seems specific to the streaming microbenchmark | [V] observation, [U] cause |

Notes:
- Target sizes 48 and 64 MB map to the same width (56.3 MB at Q8_0) because the width step is 128.
- `llc_occupancy` (about 85 MB) includes everything the process touches, not only weights.

Implication: the Stage A latency gate (t_d < 5 ms at ≤ 60 MB) passes with t_d ≈ 1.1 ms. Use 3–4 draft threads.
