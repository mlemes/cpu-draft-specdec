# b9 — bound on overlapping CPU drafting with GPU verification (2026-11-22)

Question: how much could a loop that drafts on the CPU while the GPU verifies add over llama.cpp's sequential loop?

Method (no llama.cpp change): `configs/bench.sh` measures per-pass times with llama-bench at 512-token depth;
`configs/bound.py` recovers per-step time from the measured b3 speedups (b3-json23-cpu-20261122T154807Z, b4 draft on CPU, 4 threads).

Measured:
- target pass on the RTX 3070: 14.2 ms for 1 token, 14.5 ms for 3, 15.3 ms for 5 and 7 (flat: verification is memory-bound);
- draft on the CPU: 1.72 ms per token (580 tok/s);
- per step, drafting is 16% (γ2), 24–27% (γ4) and 31–34% (γ6) of the time;
- per-step time exceeds draft + verify by 3–8 ms (sampling, draft catch-up, server bookkeeping); this overhead grows with γ and with acceptance.

Bounds (speedup over the target alone):

| γ | domain | measured | draft-ahead | oracle |
|---|---|---|---|---|
| 2 | general | 1.14 | 1.17 | 1.36 |
| 2 | pt_fin | 1.28 | 1.33 | 1.53 |
| 2 | json | 1.68 | 1.89 | 2.00 |
| 4 | general | 1.03 | 1.04 | 1.41 |
| 4 | pt_fin | 1.24 | 1.28 | 1.67 |
| 4 | json | 1.89 | 2.18 | 2.49 |
| 6 | json | 2.04 | 2.36 | 2.94 |

- draft-ahead: during verification, keep drafting assuming every token and the bonus token match the draft; useful with probability α^(γ+1) (iid approximation).
- oracle: drafting fully hidden on every step (needs drafting for several possible outcomes; the idle CPU has time for about 8 draft tokens per verification).

Conclusion: draft-ahead is worth +0.2–0.3 on JSON and only +0.01–0.05 on prose. Hiding draft time on prose needs multi-outcome drafting. The fixed per-step overhead is as large as the drafting time at γ2 and is a separate target.

Caveats: llama-bench times are isolated runs, not the server loop; hit probability assumes independent acceptances (real acceptances are correlated, which raises full-acceptance probability).
