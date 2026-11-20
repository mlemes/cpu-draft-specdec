# Stage D: adaptation economics

Asks when refreshing the student under a changing workload pays back its cost (H4). Starts only after Stage C shows a speedup somewhere. Proposal section 8.

| Experiment | Purpose |
|---|---|
| `d1-streams` | Held-out workload streams with stable, switching and recurring segments |
| `d2-update-cost` | Measured cost of each update component |
| `d3-policies` | Frozen, periodic, change-triggered and amortization policies compared on the streams |

## Cost model

- C_u = t_teacher + t_train + t_quantize + t_reload + t_warmup + t_serving_lost (GPU time taken from serving, converted to lost tokens).
- Δ = t_per_token(old draft) − t_per_token(new draft) on the new workload.
- Break-even N* = C_u / Δ tokens.

## Stop if

N* exceeds typical segment lengths: the frozen student wins. Report cumulative net savings either way.
