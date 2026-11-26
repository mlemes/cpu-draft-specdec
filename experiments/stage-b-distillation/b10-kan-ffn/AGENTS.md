# b10: Kolmogorov-Arnold feed-forward blocks at the D-S byte budget

Added 2026-11-24; design note in `docs/proposal.md` (section 8, "Design change 2026-11-24").

**Question.** At the touched bytes per token of D-S (about 56 MB at Q8_0), does replacing the SwiGLU FFN with a Kolmogorov-Arnold (KAN) block raise draft-target agreement? A secondary question: does initializing each KAN block by fitting it to the trained D-S FFN help?

**Matching.** Arms are matched on touched weight bytes per token at Q8_0, not on width (rule 2). A KAN at D-S widths would read about 9 times the FFN bytes (G + k + 1 = 9 numbers per edge) [E], so the B-spline KAN gets hidden width 256. Rational and GR-KAN arms have two matrices, so they get width 2304. `configs/budget.py` checks that every arm is within 1% of D-S.

**Arms** (`configs/arms.json`). This is a chain, and each arm changes one property from the arm before it:

| Arm | FFN | Change from previous |
|---|---|---|
| `ds-ref` | SwiGLU 512-1536-512 | b5 checkpoint; source of all non-FFN weights; not retrained |
| `ds-cont` | SwiGLU 512-1536-512 | control: everything copied from `ds-ref`, 96M more tokens |
| `ds-fresh` | SwiGLU 512-1536-512 | control: FFN re-initialized |
| `mlp-silu` | MLP 512-2304-512, SiLU | gate removed |
| `mlp-rational` | MLP 512-2304-512, learnable rational | learnable activation |
| `gr-kan` | GR-KAN 512-2304-512 | rational on the inputs of both linear maps (KAT layout) |
| `kan-bspline` | B-spline KAN 512-256-512, G=5, k=3 | per-edge splines |
| `kan-bspline-fit` | as above | initialization: per-layer fit to the D-S FFN |

Every trained arm copies attention, norms, embeddings and the V′ head from `ds-ref` and trains 96M more tokens with the b5 data mix, schedule, seed and evaluation. The FFN is re-initialized, except in `ds-cont` (copied) and `kan-bspline-fit` (fitted). The common source is needed because a layer-wise fit is only meaningful on the hidden states of the attention it was fitted against; `ds-fresh` measures what discarding the trained FFN costs.

**Method.**
1. Build the init checkpoint per arm with `src/train/ffn_init.py` (copies non-FFN tensors from `ds-ref`; `--fit` fits each KAN FFN to the `ds-ref` FFN on `ds-ref` hidden states). Run: `configs/run_b10.sh <arm>`. All arms use the same batching (4 × 4 sequences of 1024). Non-SwiGLU FFNs run under activation checkpointing while training. Without it, their peak memory per 1024-token sequence was 4.0–4.8 GiB against 2.5 GiB for SwiGLU; with it, peaks match SwiGLU and one step takes 2.7–3.4× as long [V, synthetic loss, batch 1, 2026-11-24]. In a 2-step `train.py` smoke run, `bspline_kan` reached about 5.3k tokens/s against 31.9k for the b5 SwiGLU run [V, includes start-up], so a 96M-token KAN arm may take 3–6 h instead of under 1 h [E].
2. Train each arm. Compare offline greedy agreement per domain at 96M tokens against `ds-fresh`, `mlp-silu` and `ds-cont`, with paired bootstrap 95% CIs.
3. Only if a KAN arm passes the gate: build the llama.cpp path and benchmark t_d. A B-spline block is an elementwise basis expansion (cubic B-splines on a uniform grid are sums of shifted ReLU³) followed by a normal Q8_0 matmul. A rational unit needs `ggml_mul`, `ggml_add`, `ggml_abs` and `ggml_div`, all present at the pinned commit [V]. The cost of the basis expansion on AVX2 is [U] until measured.

**Open items.** Whether Q8_0/Q4_0 quantization of spline coefficients keeps agreement is [U]. Grid range and input normalization for B-splines inside a pre-norm transformer are [U]. Rational stability uses the safe-Padé denominator 1 + |Σ b_j x^j| (KAT) [R].

**Stop if.** No KAN arm beats both `ds-fresh` and `mlp-silu` on general-prose agreement outside the CI. Then report the negative result and build no runtime path. Expected outcome [E]: neutral or negative, given Yu et al. 2024 [R].

**Added 2026-11-26.** `gr-kan` diverged (NaN at step 11): the copied post-attention RMSNorm gains (up to 191) feed inputs up to 92 into the fresh rational unit. Arms `gr-kan-n1` and `mlp-silu-n1` reset those gains to 1 (`ffn_init --reset-ffn-norm`) and form their own pair. `mlp-rational` was not run.

**Result (2026-11-26, one seed, offline greedy agreement at 96M tokens, `results/b10-20261126/`).** Every KAN arm fails the stop rule. Against `ds-fresh` on general prose: `kan-bspline` −0.021, `kan-bspline-fit` −0.011, `gr-kan-n1` −0.005 (CI includes 0). `gr-kan-n1` − `mlp-silu-n1` (same reset): −0.006 general, +0.005 pt_fin (CIs include 0), +0.008 json. Best arm: `ds-cont` (0.595). Write-up: `docs/article-kan.md`.

**Outputs.** Per-arm `metrics.jsonl` and agreement summaries in `results/<run_id>/`, byte budgets from `configs/budget.py` in each manifest.
