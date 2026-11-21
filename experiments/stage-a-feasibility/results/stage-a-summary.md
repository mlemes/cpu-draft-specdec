# Stage A summary (2026-11-21)

Machine: Ryzen 5 5500X3D (6 cores, 96 MB L3), RTX 3070 8 GB, PCIe Gen3 x16. llama.cpp `29b52d398`. Per-experiment details and raw data are in each experiment's `results/<run_id>/notes.md`.

## Measurements

| Exp | Quantity | Value | Label |
|---|---|---|---|
| a0 | resctrl counters | CAT occupancy exact; MBM within ±1% at full miss; soft L3 transition (no cliff) | [V] |
| a1 | Qwen3-8B Q4_K_M decode t_t | 12.63 / 12.88 / 13.57 ms at depth 0 / 512 / 2048 | [V] |
| a1 | verify 5 tokens (γ = 4) t_v | 14.2 ms (1.12 t_t) | [V] |
| a2 | CPU draft latency, ~56 MB touched (D-S size) | 1.14 ms/token at 3 threads, DRAM traffic 3–5% of touched bytes | [V] |
| a2 | same model with 1 L3 way (6 MB) | 2.41 ms/token: cache residency alone is worth 2.5× | [V] |
| a3 | PCIe host→device | ~5 µs for ≤ 4 KB, 245 µs for 3 MB pinned | [V] |
| a4 | Qwen3-0.6B draft, per-position α (γ = 4, greedy) | 0.63 general, 0.66 pt_fin, 0.99 json | [V] |
| a4 | 0.6B draft on the CPU | 0.25–0.47× (DRAM-bound): negative control holds | [V] |
| a4 | 0.6B draft on the GPU | 1.03–1.09× prose, 1.90× json | [V] |
| a5 | Qwen3-8B probability mass inside a 32k-token V′ | 0.960 general, 0.984 pt_fin, 0.981 json | [V] |
| a6 | llama-68m (73 MB) CPU draft for Llama-2-7B | 1.30× general, 1.03× pt_fin (greedy, γ = 4); GPU draft 1.65× / 1.32× | [V] |
| a6 | fixed CPU-draft overhead per verification cycle in llama-server | ≈ 6 ms | [V] estimate from cycle times, [U] cause |

## Break-even α for a CPU draft against Qwen3-8B

Cost model speedup = E(α, γ)·t_t / (γ·t_d + t_v + overhead), with t_t = 12.63 ms and t_v = 14.2 ms (measured at γ = 4, used for all γ [E]). α is the per-position acceptance probability.

| t_d (ms) | overhead (ms) | γ=2: 1.0× / 1.3× | γ=4: 1.0× / 1.3× | γ=6: 1.0× / 1.3× |
|---|---|---|---|---|
| 0.68 | 0 | 0.19 / 0.42 | 0.25 / 0.43 | 0.31 / 0.47 |
| 0.68 | 3 | 0.35 / 0.58 | 0.37 / 0.53 | 0.41 / 0.55 |
| 1.1 | 0 | 0.24 / 0.47 | 0.32 / 0.49 | 0.39 / 0.54 |
| 1.1 | 3 | 0.39 / 0.62 | 0.42 / 0.58 | 0.47 / 0.60 |
| 1.1 | 6 (as measured in a6) | 0.51 / 0.75 | 0.50 / 0.65 | 0.53 / 0.66 |

Reference points: the undistilled 0.6B reaches α ≈ 0.63–0.66 on prose; a5 caps α at ≈ 0.96–0.98 for V′ = 32k.

## Decision: GO, narrowed

1. **Go.** All gates pass: counters valid, D-S-sized draft is L3-resident at ~1.1 ms/token, V′ = 32k coverage ≥ 0.96, and a6 shows a real CPU-draft speedup with a 68M undistilled draft.
2. **Narrowed.** The speedup claim (H1) must be made against a GPU-draft baseline, which won in a6 (1.65× vs 1.30×). The CPU draft's contribution is only defensible if (a) the fixed CPU-path overhead is removed, or (b) the draft runs concurrently with target verification (the GPU stays busy). Stage B's first task is to locate the ≈ 6 ms (profile llama-server's speculative loop with a CPU draft) before training any model.
3. **Targets for D-S** (~50M, distilled): α ≥ 0.5 at γ = 4 for 1.0× with today's overhead; ≥ 0.45 for 1.3× once overhead is below 1 ms.
4. Portuguese is the weaker domain for off-the-shelf drafts (a6 α 0.42 vs 0.56); keep pt_fin in the distillation mix.
