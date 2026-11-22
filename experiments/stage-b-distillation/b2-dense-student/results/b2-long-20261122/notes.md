# b2 long run: kd-pruned for 96M tokens (2026-11-22)

Same as the pilot kd-pruned arm (`../b2-20261121`) except `TOKENS=96000000` (about 3.9 passes over the 24.6M scored
tokens; pt_fin about 35 passes, json about 17). Trainer with pooled gradient accumulation (`step_backward`, commit 76a89cd).
Run: `/mnt/hdd1/cpu-draft-specdec/runs/b2/kd-pruned-20261122T020931Z`. Final train loss 1.09 (pilot 1.94).

## Greedy agreement with the target (offline, 95% CI)

| tokens | general | pt_fin | json |
|---|---|---|---|
| 24M (pilot, separate run) | 0.438 [0.421, 0.456] | 0.590 [0.577, 0.602] | 0.527 [0.509, 0.545] |
| 24M (this run, mid-schedule) | 0.45 | 0.60 | 0.62 |
| 48M | 0.52 | 0.64 | 0.71 |
| 96M | **0.563** [0.544, 0.583] | **0.653** [0.640, 0.667] | **0.749** [0.732, 0.766] |
| Qwen3-0.6B, undistilled (a4) | 0.63 | 0.66 | 0.99 |

(Mid-schedule values are read from the 20M/28M and 44M/52M evaluations; see `metrics.jsonl`.)

## Reading

- Four times the training raised agreement by 0.13 (general), 0.06 (pt_fin) and 0.22 (json).
- pt_fin now matches the undistilled 0.6B within the interval; general is 0.07 below it.
- Gains flatten after about 60M tokens (general +0.02, pt_fin +0.004, json +0.02 over the last 36M), consistent with
  repeating the same 24.6M tokens; more distinct teacher-scored text is the next lever for prose.
- Under the cost model, α 0.56 on general passes the ~0.55 needed for 1.3x; the real benchmark follows (b3, 96M draft).
