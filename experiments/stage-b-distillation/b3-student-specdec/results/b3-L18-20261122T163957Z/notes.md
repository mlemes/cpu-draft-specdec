# b3, 18-layer draft (2026-11-22)

Draft: b6 (18 layers, b5 mix, 96M tokens), `/mnt/hdd1/cpu-draft-specdec/gguf/students/ds-kd-pruned-L18-96M-Q8_0.gguf`.
Same plan as the b4 repeat (`../b3-json23-cpu-20261122T154807Z`), run right after it under the same background load
(15-minute load average 3.1 before, 3.7 after). Baseline 77.1 tokens/s, 0 errors.

## Greedy speedup, 18 layers against 12 layers (b4 repeat), 100 prompts each

| γ | draft on | general 18 / 12 | pt_fin 18 / 12 | json 18 / 12 |
|---|---|---|---|---|
| 2 | CPU | 1.07 / 1.14 | 1.17 / 1.28 | 1.50 / 1.68 |
| 4 | CPU | 0.93 / 1.03 | 1.09 / 1.24 | 1.65 / 1.89 |
| 4 | GPU | 1.06 / 1.07 | 1.26 / 1.33 | 1.92 / 2.09 |
| 6 | CPU | 0.80 / 0.90 | 0.98 / 1.11 | 1.74 / 2.04 |
| 6 | GPU | 0.93 / 0.96 | 1.15 / 1.22 | 2.04 / 2.21 |

Implied alpha (18 layers): general 0.53–0.54, pt_fin 0.63–0.64, json 0.88–0.89 (12 layers: 0.51–0.52, 0.63–0.64, 0.89).
CIs about ±0.02–0.05; see `metrics.jsonl` and `src/measure/b3_summarize.py`.

## Reading

- The deeper draft is slower in every cell: +0.02 alpha on general does not pay for 1.5x the drafting work, as the cost
  model predicted (b6 notes). The loss is larger on the CPU (0.07–0.30) than on the GPU (0.01–0.17), since CPU draft
  time scales with the weights read per token.
- Conclusion: at this budget, a 12-layer draft is the better trade; extra capacity must come without extra weights
  read per token (for example a mixture of experts).
