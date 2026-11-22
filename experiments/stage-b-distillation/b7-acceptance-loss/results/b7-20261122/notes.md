# b7: total-variation loss against forward KL (2026-11-22)

Both arms continue the b5 final checkpoint for 24M tokens on the b5 mix, peak LR 1e-4, seed 1, batch 2 x 8 (4 x 4 ran
out of memory with other processes on the GPU; same effective batch in both arms). Runs:
`/mnt/hdd1/cpu-draft-specdec/runs/b2/b7-kd-20261122T173639Z`, `.../b7-tv-20261122T180006Z`.

## Offline at 24M extra tokens (95% CI)

Greedy agreement (greedy acceptance):

| arm | general | pt_fin | json |
|---|---|---|---|
| b5 start | 0.568 | 0.648 | 0.888 |
| kd | 0.582 [0.563, 0.603] | 0.652 [0.638, 0.666] | 0.888 [0.879, 0.898] |
| tv | 0.576 [0.558, 0.596] | 0.655 [0.642, 0.669] | 0.891 [0.881, 0.900] |

Overlap lower bound `tv_lower` = sum over the teacher's top-32 of min(p, q) (sampled acceptance at temperature 1):

| arm | general | pt_fin | json |
|---|---|---|---|
| kd | 0.488 [0.472, 0.505] | 0.629 [0.620, 0.638] | 0.854 [0.842, 0.864] |
| tv | **0.544** [0.528, 0.561] | **0.649** [0.640, 0.658] | **0.871** [0.860, 0.881] |

## Reading

- Greedy acceptance: no difference (the top token is already what both losses get right).
- Sampled acceptance: the TV loss adds 0.056 on general, 0.020 on pt_fin and 0.017 on json, intervals disjoint on
  general and pt_fin. Forward KL spreads probability over the teacher's tail, which costs overlap; TV optimizes overlap.
- Recommendation: KD pretraining then a short TV phase for drafts used with sampling. Speed benchmark (greedy and
  sampled at T 0.7, both arms) queued: `../../b3-student-specdec/results/b3-b7{kd,tv}-*`.
