# b5: new prose instead of repeated passes (2026-11-22)

Single change against b4 (`../../b4-json-mix/results/b4-20261122`): general2 and pt_general2 (20M new teacher-scored
tokens each, Wikipedia documents disjoint from b1) join the mix. Weights: general 0.135, general2 0.225, pt_general
0.135, pt_general2 0.225, pt_fin 0.05, json 0.08, json_extra 0.15. Each prose token is now seen about 1.1 times instead of about 3.
Run: `/mnt/hdd1/cpu-draft-specdec/runs/b2/kd-pruned-prose45json23-20261122T105044Z`.

## Greedy agreement at 96M tokens (offline, 95% CI)

| run | general | pt_fin | json |
|---|---|---|---|
| 96M baseline (JSON 0.08) | 0.563 [0.544, 0.583] | 0.653 [0.640, 0.667] | 0.749 [0.732, 0.766] |
| b4 (JSON 0.23) | 0.560 [0.541, 0.580] | 0.648 [0.635, 0.662] | 0.888 [0.877, 0.898] |
| b5 (JSON 0.23, new prose) | 0.568 [0.550, 0.588] | 0.648 [0.634, 0.662] | 0.888 [0.878, 0.898] |
| Qwen3-0.6B, undistilled (a4) | 0.63 | 0.66 | 0.99 |

General agreement along the run (b4 / b5): 16M 0.396 / 0.422, 40M 0.504 / 0.519, 64M 0.547 / 0.555, 88M 0.560 / 0.574.

## Reading

- New text gives at most +0.01 on general (within the interval) and nothing on pt_fin or json.
- Train loss over the last 20 logged steps rose from 0.91 (b4) to 0.98: b4 was partly fitting repeated text, yet that did
  not hurt held-out agreement. The prose plateau near 0.57 is not caused by data repetition.
- Remaining suspects: student capacity (53M non-embedding parameters; 0.6B reaches 0.63) and the schedule (cosine to 10%
  over 96M tokens). Next: a capacity test within the CPU-cache budget.
