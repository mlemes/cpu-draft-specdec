# a6 run 2026-11-21T03:24:08Z: findings

Target Llama-2-7B Q4_K_M (GPU, 88.5 tokens/s alone), draft llama-68m Q8_0 (73 MB; BOS id patched to 1, see `$HDD/gguf/SHA256SUMS`), same driver and plan as a4, general and pt_fin prompt sets. 1,680 requests, no errors. Draft alone on the CPU: 0.68 ms/token at 3 threads (llama-bench, 1475 ± 62 tokens/s).

| γ=4 greedy | acceptance (accepted/drafted) | implied per-position α | speedup CPU draft | speedup GPU draft |
|---|---|---|---|---|
| general | 0.31 (0.29–0.33) | 0.56 | 1.30 (1.24–1.37) | 1.65 (1.58–1.71) |
| pt_fin | 0.19 (0.17–0.21) | 0.42 | 1.03 (0.97–1.08) | 1.32 (1.27–1.37) |

Sampled (T 0.7): CPU 1.14 general, 0.95 pt_fin; GPU 1.45, 1.19.

**Decision rule** (from `../../AGENTS.md`): acceptance 0.31 at γ = 4, greedy, general, with CPU speedup 1.30 > 1 → band 0.3–0.5: thin margin; Stage B must show distillation lifts acceptance. [V]

| Finding | Label |
|---|---|
| An L3-sized off-the-shelf draft on the CPU accelerates a 7B GPU target (1.30× English greedy) | [V] |
| The same draft on the GPU is faster (1.65×). Per-cycle time from E·t_token: CPU-draft cycles 19.5–23 ms vs GPU-draft 13.7–16.7 ms; the CPU path carries a fixed ≈ 6 ms per cycle beyond γ·t_d (cycle grows only ≈ 0.7 ms per extra drafted token, matching t_d) | [V] cycle times, [U] cause |
| Portuguese acceptance is much lower (α 0.41 vs 0.56): llama-68m and Llama-2 are English-centric | [V] |

Implication: the CPU draft's own compute is not the bottleneck; the fixed CPU-path overhead per verification cycle is. Candidates [U]: draft catch-up batch on the CPU, thread-pool wake-up, CPU/GPU synchronization in llama-server's serial loop. Measuring and removing it is the first Stage B task; with zero overhead the cost model predicts ≈ 1.6× for this pair.

## Correction (2026-11-21, a7)

The "fixed ≈ 6 ms CPU-path overhead" above is wrong: it assumed t_v ≈ 12 ms, but t_v at depth is 13.2 ms. a7 decomposes the cycle: the CPU−GPU gap is the draft's per-cycle catch-up batch plus slower CPU draft tokens (0.63 vs 0.26 ms) plus 1.6 ms of thread oversubscription from the default `-tbd`. With `-tbd 4`, CPU-draft speedup rises to 1.40× general / 1.12× pt_fin (see `../a6-tbd-*/notes.md`).
