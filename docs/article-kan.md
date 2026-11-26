# Kolmogorov-Arnold networks as the feed-forward block of a CPU draft model

*Status: working article, 2026-11-26. Experiment b10 (`experiments/stage-b-distillation/b10-kan-ffn/`). Companion to `docs/article.md`, which explains the project, speculative decoding and the student model.*

**The answer in short.** No, not in this setup. At the same bytes per token, a B-spline KAN feed-forward block gives the draft *lower* agreement with the target than an ordinary block: −0.021 on general English and −0.018 on Portuguese financial text against the SwiGLU control, with 95% confidence intervals that exclude zero. Starting the KAN from a layer-by-layer imitation of the trained block helps early in training and adds +0.010 on English at the end, but it still finishes below both ordinary controls. The KAN also trains 2.3× slower per token. The group-rational KAN (GR-KAN) diverged at first because of the trained input scaling it inherits; a corrected run is <!-- GRKAN-STATUS -->.

## 1. The question

The project trains a small draft model (the *student*, D-S) that runs on the CPU and proposes tokens for Qwen3-8B on the GPU (`docs/article.md`, sections 3–5). The draft is useful only if two things hold: it agrees often with the large model, and the weights it reads for every token fit in the CPU's 96 MB L3 cache. Its budget is therefore fixed in bytes: about 56 MB per token at 8-bit precision (Q8_0).

Kolmogorov-Arnold networks (KANs) are a proposed alternative to the ordinary neural-network layer. This experiment asks whether a KAN gives the draft more agreement for the same bytes.

## 2. What a KAN changes

An ordinary feed-forward layer multiplies its input by a weight matrix and then applies one fixed nonlinear function, such as SiLU, to every output. Each connection carries a single number.

A KAN layer puts a small learnable function on every connection instead (Liu et al. 2024). In the original form, each function is a spline: a smooth curve built from G + k basis pieces, here G = 5 intervals of cubic (k = 3) pieces, plus one ordinary weight. So each connection carries G + k + 1 = 9 numbers instead of 1.

Two lighter variants keep most of the ordinary layer:

- **Rational activation.** The fixed SiLU is replaced by a learnable ratio of two polynomials, shared by groups of channels. This costs a few dozen numbers per layer.
- **Group-rational KAN (GR-KAN).** From the Kolmogorov-Arnold Transformer (Yang and Wang, ICLR 2025): a learnable rational function is applied to the inputs of both matrices of the feed-forward block. This is the KAN variant with the strongest published transformer results.

An independent comparison that matched parameters and compute found KANs worse than ordinary layers on language and vision tasks, and better only on fitting symbolic formulas (Yu et al. 2024). The expected outcome here was therefore neutral or negative.

## 3. A fair comparison: bytes, not widths

A KAN with the same layer widths as the student would read about 9 times as many feed-forward bytes per token and would no longer fit the cache. A KAN with the same bytes must be narrower. Every variant here is matched on bytes read per token, not on width:

| Variant | Feed-forward block | Weights read per layer | Total per token (Q8_0) |
|---|---|---|---|
| D-S (SwiGLU) | 512 → 1536 → 512, three matrices | 2,359,296 | 56.3 MB |
| Two-matrix MLP | 512 → 2304 → 512 | 2,359,296 | 56.3 MB |
| Rational MLP, GR-KAN | 512 → 2304 → 512 plus rational coefficients | 2,359,376 / 2,359,456 | 56.3 MB |
| B-spline KAN | 512 → 256 → 512, 9 numbers per connection | 2,359,296 | 56.3 MB |

The rest of the model is identical: attention, embeddings and the output head over the 32,768-token draft vocabulary. The counts come from the tensor layout (`configs/budget.py`). Only the feed-forward block (about half of the per-token bytes) changes.

## 4. The variants

Each variant changes one property of the one before it, so a difference can be attributed:

| Arm | Change |
|---|---|
| `ds-cont` | The trained D-S, trained further. Control for the extra training. |
| `ds-fresh` | D-S with its feed-forward block reset to random weights. Control for discarding the trained block. |
| `mlp-silu` | Gate removed: two matrices instead of three, wider. |
| `mlp-rational` | SiLU replaced by a learnable rational function. |
| `gr-kan` | Rational function on the inputs of both matrices (KAT layout). |
| `kan-bspline` | Per-connection splines (a true KAN). |
| `kan-bspline-fit` | The same KAN, but each layer is first fitted to imitate the trained D-S feed-forward block. |

The last arm is the second idea from the original question: a KAN made of functions that approximate the student's layers. Fitting the layers one by one and stopping there would not work, because each layer's small error changes the input of the next. Here the fit is only a starting point, followed by the same training as the other arms.

## 5. Setup

- **Starting point.** All arms start from the trained b5 student (96M tokens of distillation; `docs/article.md` section 6.1) and keep its attention, embeddings and output head. Everything except `ds-cont` gets a new feed-forward block.
- **Training.** 96M more tokens each, with forward-KL distillation from the target's top-32 probabilities, the b5 data mix, learning rate 6e-4 with cosine decay, 16 sequences of 1024 tokens per step, seed 0, `torch.compile`. Equal tokens for every arm.
- **Metric.** Offline greedy agreement: on 100 held-out prompts per domain, the fraction of positions where the student's most likely next token equals the target's. This tracks the acceptance rate of greedy speculative decoding (`docs/article.md` section 6). Each arm is scored on the same prompts, so differences are paired: their 95% confidence intervals come from resampling prompts (2000 draws, positions-weighted) with `configs/analyze.py`.
- **Not measured.** Draft speed on the CPU. llama.cpp has no graph for these blocks, and the plan builds one only if a KAN variant wins on agreement.

## 6. Results

### 6.1 Agreement after 96M tokens

Offline greedy agreement per domain, with 95% confidence intervals over prompts. The b5 student, before the extra 96M tokens, is shown for reference.

| Arm | General English | Portuguese financial | JSON |
|---|---|---|---|
| b5 student (`ds-ref`, start point) | 0.569 [0.551, 0.589] | 0.648 [0.634, 0.663] | 0.888 [0.877, 0.898] |
| `ds-fresh` (SwiGLU, new block) | **0.585** [0.567, 0.606] | **0.686** [0.671, 0.701] | **0.885** [0.876, 0.893] |
| `mlp-silu` (two-matrix MLP) | 0.581 [0.562, 0.602] | 0.681 [0.666, 0.697] | 0.874 [0.866, 0.882] |
| `kan-bspline` (B-spline KAN) | 0.564 [0.545, 0.585] | 0.668 [0.653, 0.685] | 0.879 [0.869, 0.889] |
| `kan-bspline-fit` (KAN, fitted start) | 0.574 [0.555, 0.595] | 0.663 [0.649, 0.677] | 0.868 [0.860, 0.877] |
<!-- MORE-ARMS -->

The intervals of different arms overlap, because the prompts themselves vary a lot. The paired comparison in the next table removes that variation: every arm is scored on the same prompts, and the interval is computed on the per-prompt differences.

### 6.2 Paired differences

| Comparison | General English | Portuguese financial | JSON |
|---|---|---|---|
| `mlp-silu` − `ds-fresh` (gate removed) | −0.005 [−0.010, +0.001] | −0.005 [−0.011, +0.001] | −0.011 [−0.017, −0.006] |
| `kan-bspline` − `mlp-silu` (splines added) | −0.017 [−0.023, −0.010] | −0.013 [−0.019, −0.007] | +0.005 [−0.000, +0.010] |
| `kan-bspline` − `ds-fresh` | −0.021 [−0.028, −0.014] | −0.018 [−0.024, −0.012] | −0.006 [−0.011, −0.001] |
| `kan-bspline-fit` − `kan-bspline` (fitted start) | +0.010 [+0.003, +0.016] | −0.005 [−0.013, +0.003] | −0.011 [−0.016, −0.006] |
| `kan-bspline-fit` − `ds-fresh` | −0.011 [−0.018, −0.004] | −0.023 [−0.030, −0.016] | −0.017 [−0.022, −0.012] |
| `kan-bspline-fit` − `mlp-silu` | −0.007 [−0.012, −0.001] | −0.018 [−0.025, −0.011] | −0.006 [−0.010, −0.002] |
<!-- MORE-PAIRS -->

For scale: in the speed model of `docs/article.md` section 3.3 (γ = 4, draft 1.1 ms per token, verification 14.2 ms, 2–3 ms overhead), raising α from 0.57 to 0.58 raises the expected tokens per cycle from 2.186 to 2.225, about 1.8% more speed [E]. The KAN's loss of 0.02 is small in absolute terms but consistent, and it comes on top of the slower training.

### 6.3 How agreement develops

General-English agreement during training (measured on the same prompts every 4M tokens):

| Tokens | `ds-fresh` | `mlp-silu` | `kan-bspline` | `kan-bspline-fit` |
|---|---|---|---|---|
| 4M | 0.389 | 0.444 | 0.382 | **0.505** |
| 20M | 0.514 | 0.518 | 0.496 | **0.526** |
| 52M | **0.562** | 0.555 | 0.543 | 0.554 |
| 92M | 0.585 | **0.588** | 0.566 | 0.577 |

The fitted KAN starts far ahead, since it already imitates the trained block, but the lead is gone by about 30M tokens. The ordinary blocks catch up and pass it. The plain KAN trails the ordinary blocks by about 0.02 for the whole run.

### 6.4 How well a KAN can imitate the trained block

Before training, each `kan-bspline-fit` layer was fitted for 1954 steps (two passes over 4M tokens of teacher data) to reproduce the trained SwiGLU block's output on the same inputs. The remaining error, as a fraction of the output's variance, measured on batches not used for fitting:

| Layers | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Relative error after fit | 0.25 | 0.20 | 0.23 | 0.31 | 0.38 | 0.39 | 0.33 | 0.33 | 0.37 | 0.36 | 0.45 | 0.22 |

So a byte-matched KAN, 256 units wide, captures about 55–80% of what the 1536-unit SwiGLU block computes. That is the head start seen in section 6.3, and also its limit: the remaining error compounds through the layers, and further training does not close the gap.

### 6.5 The rational variants

<!-- RATIONAL -->

### 6.6 Training cost

| Block | Training speed (tokens/s, RTX 3070) | 96M tokens |
|---|---|---|
| SwiGLU, two-matrix MLP | about 32,000 | about 50 min |
| B-spline KAN | about 14,200 | about 1 h 55 min |

Without activation checkpointing (recomputing the block in the backward pass instead of storing its intermediate values), the rational and KAN blocks did not fit in 8 GB even at one sequence per step. With it, their peak memory matches SwiGLU, and `torch.compile` raises the KAN from 4,700 to 14,000–21,000 tokens/s. On the CPU, where the draft runs, the KAN's basis evaluation is extra work per token whose cost is unmeasured.

## 7. Reading the results

- **The B-spline KAN fails the experiment's stop rule.** The rule asked for a KAN arm to beat both the SwiGLU and the two-matrix MLP controls on general English, outside the confidence interval. Both KAN arms are below both controls.
- **Imitating the trained block is a good start, not a good model.** The fitted start gives +0.12 at 4M tokens and +0.010 at the end on English, but it is worse on JSON (−0.011), and it does not reach the ordinary blocks.
- **The result agrees with the independent comparison** of Yu et al. (2024): at matched parameters, KANs lose to ordinary layers outside symbolic-formula fitting. Here the comparison is matched in runtime bytes, which is what matters for a cache-resident draft.
- **Removing the gate costs a little** (−0.005 on prose, −0.011 on JSON), so the gated SwiGLU block is the right baseline for the student.
- **Training a fresh block for 96M more tokens helps** (`ds-fresh` against the b5 start: +0.016 on English, +0.038 on Portuguese financial text). Part of that is simply more training; <!-- DSCONT --> separates the two.

## 8. Limitations

- **One seed per arm.** The paired intervals cover prompt variation, not training-run variation.
- **Agreement, not speed.** No KAN draft ran inside llama.cpp, so there is no end-to-end speed. Because the KAN loses on agreement, the speed test was not built.
- **Inherited input scaling.** Each fresh block receives its input through the b5 student's trained normalization gains, which reach 191 in the last layer. In the trained KAN, 0.5–8.5% of the spline inputs fall outside the spline grid [−2, 2] in layers 0–10, and 37% in layer 11; there, only the block's ordinary weight path is active. A grid sized to the inputs might narrow the gap.
- **Untuned KAN settings.** One grid size (G = 5), one grid range, and one fitting schedule were tried.
- **Scoring precision.** These scores were computed in float32 on the CPU. The training-time scores, in bfloat16 on the GPU, differ by about 0.001 (0.5693 against 0.5679 for the b5 student on English).

## 9. What next

<!-- NEXT -->


## References

- Liu et al. 2024, KAN: Kolmogorov-Arnold Networks. https://arxiv.org/abs/2404.19756
- Yang and Wang, Kolmogorov-Arnold Transformer (ICLR 2025). https://arxiv.org/abs/2409.10594
- Yu, Yu and Wang 2024, KAN or MLP: A Fairer Comparison. https://arxiv.org/abs/2407.16674
- Li 2024, Kolmogorov-Arnold Networks are Radial Basis Function Networks (FastKAN). https://arxiv.org/abs/2405.06721
