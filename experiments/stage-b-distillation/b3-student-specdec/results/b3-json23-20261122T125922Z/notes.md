# b3, b4 student as a draft (2026-11-22)

Draft: kd-pruned, 96M tokens, JSON share 0.23 (b4, run `kd-pruned-json23-20261122T041545Z`), exported to
`/mnt/hdd1/cpu-draft-specdec/gguf/students/ds-kd-pruned-json23-96M-Q8_0.gguf`. Same driver, target, prompts and
settings as `../b3-96M-20261122T050950Z`. Plan `none:0:all,cpu:2:all,cpu:3:all,cpu:4:all,gpu:4:all,cpu:6:30,gpu:6:30`.
Baseline 75.1 tokens/s (96M run: 77.5). 0 errors. Table from `src/measure/b3_summarize.py metrics.jsonl`.

**Load check:** an unrelated process (one core, about 1.1 GB VRAM) ran during this measurement. A repeat
(`../b3-json23-cpu-20261122T154807Z`, all rows 100 prompts, same load; 15-minute load average 1.9 before and 3.1 after)
reproduced every CPU row within 0.06 and the GPU rows within 0.08 (slightly lower), so the numbers stand.

## Greedy speedup (95% CI by bootstrap over prompts)

| γ | draft on | prompts | general | pt_fin | json |
|---|---|---|---|---|---|
| 2 | CPU | 100 | 1.14 (1.12–1.16) | 1.27 (1.25–1.30) | 1.68 (1.66–1.70) |
| 3 | CPU | 100 | 1.09 (1.06–1.12) | 1.28 (1.24–1.32) | 1.89 (1.85–1.93) |
| 4 | CPU | 100 | 0.97 (0.95–1.00) | 1.22 (1.18–1.26) | 1.94 (1.90–1.98) |
| 4 | GPU | 100 | 1.09 (1.06–1.13) | 1.36 (1.33–1.40) | **2.17 (2.12–2.21)** |
| 6 | CPU | 30 | 0.88 (0.84–0.93) | 1.12 (1.06–1.18) | 2.05 (1.95–2.17) |
| 6 | GPU | 30 | 0.99 (0.94–1.05) | 1.26 (1.19–1.32) | **2.28 (2.17–2.41)** |

Implied per-position alpha: general 0.51–0.52, pt_fin 0.63–0.64, json 0.89–0.90 (offline b4: 0.560, 0.648, 0.888).

## Reading

- JSON: 1.43x to 1.94x (CPU, best γ in the clean range), 1.59x to 2.17x (GPU, γ 4) against the 96M 8%-JSON draft.
  Qwen3-0.6B as GPU draft at γ 4: 1.90x. The 50M draft now beats it on all three domains.
- JSON implied alpha equals the offline measure (0.89); at that rate the best γ is 6 or more.
- Prose alpha equals the 96M draft's, as offline predicted; prose speed differences against that run come from the
  CPU load noted above.

## Repeat (all γ at 100 prompts)

| γ | draft on | general | pt_fin | json |
|---|---|---|---|---|
| 2 | CPU | 1.14 (1.12–1.17) | 1.28 (1.26–1.30) | 1.68 (1.66–1.70) |
| 4 | CPU | 1.03 (0.99–1.06) | 1.24 (1.21–1.27) | 1.89 (1.85–1.93) |
| 4 | GPU | 1.07 (1.05–1.11) | 1.33 (1.30–1.37) | 2.09 (2.05–2.13) |
| 6 | CPU | 0.90 (0.87–0.93) | 1.11 (1.07–1.15) | **2.04 (1.98–2.10)** |
| 6 | GPU | 0.96 (0.93–0.99) | 1.22 (1.17–1.26) | **2.21 (2.14–2.28)** |

Baseline 77.2 tokens/s. The CPU-to-GPU gap on json (0.17–0.20) is real: at alpha 0.89 each cycle drafts and keeps more
tokens, so the CPU's slower drafting is a larger share of the cycle.
