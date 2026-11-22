# b3: distilled student as a real llama.cpp draft (2026-11-22)

Target Qwen3-8B Q4_K_M on the RTX 3070; draft = kd-pruned student after 24M tokens (b2 pilot), exported with
`src/models/export_student_gguf.py` and quantized to Q8_0 (139 MB file; V′ head 32,768 rows), patched llama.cpp
(`src/patches/llama.cpp-vprime.patch`). Driver `src/measure/a4_reference_draft.py`, same prompts, plan and settings as
the Qwen3-0.6B reference run (`a4-20261121T015524Z`): `--spec-type draft-simple`, fixed γ, 64 tokens, 100 prompts per domain
at γ = 4 and 30 at γ = 2, 6; CPU draft `-td 4 -tbd 4`. 2,520 requests, 0 errors. Baseline: target alone, 76 tokens/s.
Speedup = per-prompt tokens/s ratio against the same prompt on the target alone; 95% CI by bootstrap over prompts.

## Greedy speedup

| γ | draft on | general | pt_fin | json |
|---|---|---|---|---|
| 2 | CPU | 1.04 (1.00–1.07) | **1.27 (1.23–1.31)** | 1.07 (1.03–1.11) |
| 2 | GPU | 1.05 (1.01–1.09) | 1.26 (1.23–1.30) | 1.11 (1.07–1.15) |
| 4 | CPU | 0.92 (0.89–0.94) | 1.18 (1.15–1.21) | 1.02 (0.99–1.05) |
| 4 | GPU | 0.96 (0.93–0.98) | 1.26 (1.23–1.29) | 1.09 (1.06–1.12) |
| 6 | CPU | 0.78 (0.74–0.82) | 1.07 (1.02–1.12) | 0.89 (0.84–0.95) |
| 6 | GPU | 0.81 (0.77–0.86) | 1.11 (1.06–1.16) | 0.95 (0.90–1.01) |

Sampled (T 0.7) speedups are 0.01–0.10 lower; see `summary.json`.

Reference, Qwen3-0.6B draft on the GPU at γ = 4 (greedy): general 1.03, pt_fin 1.09, json 1.90; on the CPU 0.25–0.47.

## Offline acceptance predicts online acceptance

Server acceptance at γ = 4 (accepted / drafted) converts to a per-position rate through E = 1 + γ·acc and
E = (1 − α^(γ+1)) / (1 − α):

| domain | server accepted/drafted | implied per-position α | offline greedy agreement (b2) |
|---|---|---|---|
| general | 0.185 | 0.43 | 0.44 (0.42–0.46) |
| pt_fin | 0.331 | 0.60 | 0.59 (0.58–0.60) |
| json | 0.241 | 0.51 | 0.53 (0.51–0.55) |

The offline evaluator is a reliable proxy, so later training runs can be compared without full speed runs.

## Findings

- First real speedup from the cache-sized CPU draft: 1.27× on Portuguese financial text at γ = 2, equal to the same draft on the GPU.
- At γ = 2 the CPU draft matches the GPU draft in every domain (intervals overlap); the GPU pulls ahead only at γ ≥ 4,
  where the CPU's slower per-token drafting accumulates. A CPU draft at γ = 2 therefore costs no speed and uses no GPU memory.
- The 50M student beats the 8x larger Qwen3-0.6B on prose (pt_fin 1.27 vs 1.09, general 1.05 vs 1.03) but not on JSON
  (1.11 vs 1.90), where the 0.6B agrees with the target almost perfectly (α 0.99 vs 0.51). JSON had 8% of the training mix.
- On general English the student is near break-even (1.04–1.05 at γ = 2, below 1.0 at γ ≥ 4), as the cost model predicted for α ≈ 0.44.
- Shorter blocks win at this acceptance level: γ = 2 is best everywhere.

## Next

Longer training (96M tokens, running), then rerun this benchmark; more JSON in the mix if structured output matters.
