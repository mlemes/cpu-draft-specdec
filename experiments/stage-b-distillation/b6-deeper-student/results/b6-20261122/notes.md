# b6: 18 layers instead of 12 (2026-11-22)

Single change against b5 (`../../b5-more-prose/results/b5-20261122`): depth 18, pruned from Qwen3-0.6B with evenly
spaced source layers. Batch 2 x accumulation 8 (4 x 4 ran out of memory with other processes holding about 1.7 GB of
VRAM); same effective batch, tokens, seed, schedule and eval. Run:
`/mnt/hdd1/cpu-draft-specdec/runs/b2/kd-pruned-L18-20261122T135232Z`. Draft GGUF
`/mnt/hdd1/cpu-draft-specdec/gguf/students/ds-kd-pruned-L18-96M-Q8_0.gguf` (165 MB file; per-token weights about 75 MB).

## Greedy agreement at 96M tokens (offline, 95% CI)

| run | general | pt_fin | json |
|---|---|---|---|
| b5, 12 layers | 0.568 [0.550, 0.588] | 0.648 [0.634, 0.662] | 0.888 [0.878, 0.898] |
| b6, 18 layers | 0.585 [0.565, 0.606] | 0.654 [0.641, 0.667] | 0.882 [0.873, 0.891] |
| Qwen3-0.6B, undistilled (a4) | 0.63 | 0.66 | 0.99 |

## Reading

- Depth adds +0.017 on general (intervals overlap), nothing on pt_fin or json. General was still rising at the end
  (+0.01 over the last 12M tokens); the others were flat.
- Cost model at γ 2: alpha 0.568 to 0.585 raises tokens per cycle by about 2%, while 1.5x the draft layers adds about
  0.5 ms per drafted token (about 1 ms per cycle of about 19 ms, -5%). Expected net: slightly slower. The speed
  benchmark (b3-L18, queued after the b4 CPU rerun under the same load) checks this.
- Capacity is not the main limit at this budget: the 6-point prose gap to Qwen3-0.6B needs more than 1.5x depth.
