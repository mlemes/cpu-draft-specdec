# b1 pilot: teacher top-32 scoring (2026-11-21)

Teacher: Qwen3-8B-Q4_K_M (sha256 prefix d98cdcbd03e1), RTX 3070, `teacher_score` seq-len 1024, k 32, threads 5.
Script: `configs/run_b1_pilot.sh`. Outputs: `/mnt/hdd1/cpu-draft-specdec/teacher/<domain>/d98cdcbd03e1/` (4.5 GB total).
Wall time 184 min for 24.6M tokens.

| domain | sequences | tokens | tokens/s | mean tail mass | p90 tail | mean top-1 p |
|---|---|---|---|---|---|---|
| general | 11,718 | 11,999,232 | 2,241 | 0.041 | 0.127 | 0.643 |
| pt_general | 11,718 | 11,999,232 | 2,262 | 0.043 | 0.150 | 0.707 |
| pt_fin | 133 | 136,192 | 2,219 | 0.045 | 0.148 | 0.693 |
| json | 450 | 460,800 | 2,152 | 0.005 | 0.000 | 0.904 |

Tail and top-1 statistics are from shard 0 of each domain. Top-32 covers about 96% of the teacher's
probability mass on prose and 99.5% on JSON, so the tail-bin KD loss loses little information.
pt_fin is small (Copom minutes 232-261 only); it is upweighted to 0.05 in b2 and will repeat across epochs.
