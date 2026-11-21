# a4 run 2026-11-21T01:55:24Z: findings

Target Qwen3-8B Q4_K_M (GPU), draft Qwen3-0.6B Q8_0, llama-server `--spec-type draft-simple`, fixed γ (`n-min = n-max`, `p-min 0`), 64 generated tokens, 100 prompts per domain (γ = 4) or 30 (γ = 2, 6), greedy and sampled (T 0.7). 2,520 requests, no errors. Acceptance = accepted / drafted tokens. Speedup = per-prompt tokens/s ratio versus the target alone (77.5 greedy, 78.1 sampled tokens/s); 95% CIs by bootstrap over prompts.

| γ=4 | acceptance greedy | acceptance sampled | speedup CPU draft (greedy) | speedup GPU draft (greedy) |
|---|---|---|---|---|
| general | 0.37 (0.35–0.39) | 0.30 | 0.25 (0.24–0.25) | 1.03 (1.00–1.07) |
| pt_fin | 0.42 (0.39–0.44) | 0.36 | 0.26 (0.25–0.27) | 1.09 (1.06–1.13) |
| json | 0.97 (0.96–0.98) | 0.97 | 0.47 (0.46–0.47) | 1.90 (1.88–1.92) |

| Finding | Label |
|---|---|
| A 0.6B draft on the CPU slows decoding 2–5× in every cell: the draft streams ~0.6 GB per token from DRAM (a2: ~12–20 ms/token at that size) | [V] |
| The same draft on the GPU gives 1.0–1.1× on prose and 1.9–2.25× on json (γ = 6) | [V] |
| Per-drafted-token acceptance falls with γ on prose (general greedy 0.55 / 0.37 / 0.28 at γ = 2 / 4 / 6) | [V] |
| JSON outputs are near-deterministic for both models (acceptance 0.95–0.99); cause likely structure copied from the schema in the prompt | [V] measured, [U] cause |
| Sampled acceptance is 0.05–0.08 below greedy on prose | [V] |

Implication: the negative control holds (H1 needs an L3-resident draft, not a small off-the-shelf one). Prose acceptance of 0.37–0.42 at γ = 4 for a 0.6B undistilled draft sets the bar a ~50M distilled D-S must approach.
