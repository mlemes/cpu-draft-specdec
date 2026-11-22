# b3, b4 student as a draft (2026-11-22)

Draft: kd-pruned, 96M tokens, JSON share 0.23 (b4, run `kd-pruned-json23-20261122T041545Z`), exported to
`/mnt/hdd1/cpu-draft-specdec/gguf/students/ds-kd-pruned-json23-96M-Q8_0.gguf`. Same driver, target, prompts and
settings as `../b3-96M-20261122T050950Z`. Plan `none:0:all,cpu:2:all,cpu:3:all,cpu:4:all,gpu:4:all,cpu:6:30,gpu:6:30`.
Baseline 75.1 tokens/s (96M run: 77.5). 0 errors. Table from `src/measure/b3_summarize.py metrics.jsonl`.

**Caveat:** an editor and several agent sessions started on the machine during this run (15-minute load average 4.6
afterwards). The CPU draft uses 4 of 6 cores, so CPU-draft speedups here are likely understated: at the same implied
alpha as the 96M run on prose, the CPU-to-GPU gap at γ 4 grew from 0.05 to 0.12. GPU-draft rows and implied alpha are
not affected. A clean rerun of the CPU rows is queued.

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
