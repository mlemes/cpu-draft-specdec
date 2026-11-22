# b3, 96M-token draft (2026-11-22)

Same driver, target, prompts and settings as `../b3-20261122T012740Z`; draft = kd-pruned after 96M tokens
(`/mnt/hdd1/cpu-draft-specdec/gguf/students/ds-kd-pruned-96M-Q8_0.gguf`, run `kd-pruned-20261122T020931Z`).
Plan `none:0:all,cpu:2:all,gpu:2:all,cpu:4:all,gpu:4:all,cpu:3:30,gpu:3:30,cpu:6:30,gpu:6:30`; 64 tokens; CPU
draft 4 threads. Baseline: target alone, 77.5 tokens/s. Table from `src/measure/b3_summarize.py metrics.jsonl`.

## Greedy speedup (95% CI by bootstrap over prompts)

| γ | draft on | prompts | general | pt_fin | json |
|---|---|---|---|---|---|
| 2 | CPU | 100 | **1.18 (1.16–1.21)** | 1.31 (1.29–1.33) | 1.43 (1.40–1.46) |
| 2 | GPU | 100 | 1.21 (1.18–1.24) | 1.36 (1.33–1.38) | 1.48 (1.45–1.51) |
| 3 | CPU | 30 | 1.12 (1.08–1.16) | **1.32 (1.27–1.37)** | 1.47 (1.41–1.53) |
| 3 | GPU | 30 | 1.15 (1.11–1.19) | 1.37 (1.32–1.42) | 1.52 (1.45–1.58) |
| 4 | CPU | 100 | 1.06 (1.03–1.09) | 1.28 (1.25–1.31) | **1.50 (1.46–1.55)** |
| 4 | GPU | 100 | 1.11 (1.08–1.14) | 1.34 (1.31–1.38) | 1.59 (1.53–1.64) |
| 6 | CPU | 30 | 0.95 (0.90–1.00) | 1.14 (1.09–1.20) | 1.40 (1.31–1.48) |
| 6 | GPU | 30 | 1.00 (0.95–1.06) | 1.22 (1.16–1.28) | 1.50 (1.41–1.59) |

Sampled (T 0.7) speedups: 0.05–0.08 lower on prose, equal on json; see `summary.json`.

## Implied per-position alpha (server accepted/drafted)

| domain | implied alpha (all γ) | offline greedy agreement (b2 96M) |
|---|---|---|
| general | 0.51–0.53 | 0.563 [0.544, 0.583] |
| pt_fin | 0.63–0.64 | 0.653 [0.640, 0.667] |
| json | 0.70–0.72 | 0.749 [0.732, 0.766] |

## Reading

- Against the 24M draft (best CPU): general 1.04 to 1.18, pt_fin 1.27 to 1.32, json 1.07 to 1.50.
- Optimal γ grows with alpha: 2 for general, 3 for pt_fin, 4 for json.
- Implied alpha is 0.01–0.03 below offline agreement; the offline measure still ranks drafts correctly.
- CPU draft now 0.03–0.10 behind the GPU draft (24M draft: indistinguishable at γ 2). With higher acceptance each cycle
  carries more tokens, so the CPU's slower drafting is a larger share of cycle time.
- Qwen3-0.6B on the GPU at γ 4 (a4): general 1.03, pt_fin 1.09, json 1.90. The 50M CPU draft now beats it on prose
  by 0.15–0.23.
