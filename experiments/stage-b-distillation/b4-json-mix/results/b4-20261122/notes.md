# b4: JSON share 0.08 to 0.23 (2026-11-22)

Single change against the 96M-token kd-pruned run (`../../b2-dense-student/results/b2-long-20261122`): the mix adds
`json_extra` (9,501 Hermes function-calling and agentic JSON conversations, 40.7M chars, teacher-scored with
`--parse-special`). Weights: general 0.36, pt_general 0.36, pt_fin 0.05, json 0.08, json_extra 0.15.
Run: `/mnt/hdd1/cpu-draft-specdec/runs/b2/kd-pruned-json23-20261122T041545Z`.

## Greedy agreement with the target at 96M tokens (offline, 95% CI)

| run | general | pt_fin | json |
|---|---|---|---|
| 96M baseline (JSON 0.08) | 0.563 [0.544, 0.583] | 0.653 [0.640, 0.667] | 0.749 [0.732, 0.766] |
| b4 (JSON 0.23) | 0.560 [0.541, 0.580] | 0.648 [0.635, 0.662] | **0.888** [0.877, 0.898] |
| Qwen3-0.6B, undistilled (a4) | 0.63 | 0.66 | 0.99 |

## Reading

- JSON +0.14, intervals disjoint. Prose unchanged within the intervals (general -0.003, pt_fin -0.005).
- Prose sampling weights fell (general and pt_general 0.435 to 0.36 each), but those domains were repeating the same
  24.6M tokens and had saturated, so fewer repeats cost nothing.
- JSON still rising at the end (+0.01 over the last 8M tokens); prose flat since about 60M.
- Under the cost model, alpha 0.89 on JSON gives about 1.7x at gamma 2 and more at gamma 4 (the model ran about 0.08 optimistic in b3); measured benchmark of this student is pending.
- b4 dominates the 96M baseline: future runs should start from the b4 mix.
