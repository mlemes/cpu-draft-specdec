# a7 2026-11-21: per-cycle decomposition (Llama-2-7B Q4_K_M target on GPU, llama-68m Q8_0 draft, γ = 4)

## How one draft-simple cycle works (llama.cpp 29b52d398, `common/speculative.cpp:301-480`)

Serial loop, nothing overlaps:
1. target verifies a batch of γ+1 tokens (GPU);
2. `process()`: the draft decodes **the same γ+1-token batch** to keep its KV cache in step (one batch decode on the draft device);
3. `draft()`: γ single-token decodes on the draft device, each followed by `common_sampler_sample` over the full vocabulary.

## Component costs (llama-bench, depth 350, r = 10) [V]

| Component | GPU | CPU 4 threads | CPU 1 thread |
|---|---|---|---|
| target verify, 5 tokens (t_v) | 13.22 ms | — | — |
| target decode, 1 token (t_t) | 11.51 ms | — | — |
| draft batch, 5 tokens | 0.33 ms | 1.26 ms | 3.98 ms |
| draft decode, 1 token | 0.26 ms | 0.63 ms | 2.00 ms |

## Measured cycle times (configs/probe.py) [V]

| Variant | cycle ms | t_v + batch + γ·token | residual |
|---|---|---|---|
| GPU draft | 16.32 | 14.58 | 1.74 |
| CPU `-td 4` (as in a4/a6; `-tbd` default = all cores) | 20.86 | 16.98 | 3.88 |
| CPU `-td 4 -tbd 4` | 19.25 | 16.98 | 2.27 |
| CPU `-td 4 -tbd 4`, no polling | 19.15 | 16.98 | 2.17 |
| CPU `-td 1 -tbd 1` | 27.40 | 25.18 | 2.22 |
| CPU `-td 3 -t 1`, pinned cores 1-3 | 20.27 | | |

## Findings

| Finding | Label |
|---|---|
| There is no mysterious fixed ≈ 6 ms CPU overhead. a6's estimate assumed t_v ≈ 12 ms; the measured t_v at depth is 13.2 ms. | [V] (corrects a6) |
| The component model explains every variant to within ~2 ms; the ~2 ms residual is common to CPU and GPU drafts (server loop, sampling, bookkeeping) | [V] |
| CPU vs GPU draft gap per cycle (2.9 ms with `-tbd 4`) = draft catch-up batch (+0.93 ms) + γ draft tokens (+1.48 ms) + ~0.5 ms | [V] |
| Default `-tbd` (all 6 cores) oversubscribes the CPU during the catch-up batch: +1.6 ms per cycle. a4/a6 CPU numbers carry this penalty; the driver now passes `-tbd = -td` | [V] |
| At 68M parameters the RTX 3070 decodes the draft faster than the CPU (0.26 vs 0.63 ms/token), so in a serial loop a GPU draft always wins | [V] |

## Implication for H1

In llama.cpp's serial speculative loop the draft device matters only through draft time, and the GPU is faster even for an L3-sized draft. A CPU draft can only beat a GPU draft when it **runs concurrently** with target verification (draft the next block on the CPU while the GPU verifies the current one, as in PEARL-style pipelined speculation), or when the GPU is busy with other work. The project's speedup claim therefore needs an asynchronous draft/verify loop; that is the first Stage B engineering task. Upper bound with perfect overlap at γ = 4: cycle ≈ max(t_v, γ·t_d + batch) ≈ 13.2 ms instead of 16.3 (GPU draft) [E].
