# Stage B: distillation and cache behavior

Trains the students and measures acceptance, cache residency and standalone quality. Starts only after the Stage A gate passes. Proposal sections 5, 6 and 8.

Teacher data and training never share the GPU: run `b1` first, then train with the GPU dedicated.

| Experiment | Purpose |
|---|---|
| `b1-teacher-data` | Top-32 teacher records from the deployed target |
| `b2-dense-student` | D-S: hard labels versus forward-KL KD versus TV KD; pruned versus scratch initialization |
| `b3-moe-student` | M upcycled from D-S, with and without sticky routing |
| `b4-dense-large-control` | D-L, dense with the same stored bytes as M |
| `b5-cache-behavior` | Routing churn, expert reuse, DRAM traffic and CAT sweeps for each student |
| `b6-standalone-quality` | Each student evaluated on its own |

## Controls

- Equal token budgets (300M tokens per arm). M versus D-S continued for the same number of tokens.
- Fixed V′, context lengths 256 and 2048, thread count, and quantization per comparison.

## Stop if

- D-S α is below the break-even α for its measured t_d minus 0.05: stop the neural CPU draft and narrow to a GPU draft or a prompt-lookup hybrid.
- H2 is rejected if M's Δα ≤ 0.02 within the CI, or if its DRAM bytes per token erase its latency advantage.
