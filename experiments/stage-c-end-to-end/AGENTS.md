# Stage C: end-to-end speculative decoding

Measures real speedup against every baseline. Starts only after the Stage B gate passes. Proposal sections 7, 8 and 9.

| Experiment | Purpose |
|---|---|
| `c1-correctness` | Prove the greedy and stochastic procedures preserve the stated target distribution |
| `c2-baselines` | GPU target alone, GPU-resident draft, n-gram or prompt lookup, standalone CPU student |
| `c3-e2e-sweep` | CPU draft with GPU verification over a chosen subset of settings |

`c1` must pass before `c3` results are reported.

## Stop if

The speedup's 95% CI lower bound is ≤ 1.0 for every draft. Report the negative result with its round-time breakdown.
