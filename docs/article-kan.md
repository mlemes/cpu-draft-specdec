# Kolmogorov-Arnold networks as the feed-forward block of a CPU draft model

*Status: working article, 2026-11-26; three-seed update 2026-11-27. These are development results: the evaluation prompts were seen by the agents that ran the experiment, and an independent second pass on sealed prompts is planned before publication (`docs/proposal.md`, design change 2026-11-27). Experiment b10 (`experiments/stage-b-distillation/b10-kan-ffn/`). Companion to `docs/article.md`, which explains the project, speculative decoding and the student model.*

**The answer in short.** No, not in this setup. At the same bytes per token, no KAN variant beats the ordinary feed-forward block on general-English agreement, the experiment's decision metric (section 5).

- **B-spline KAN** (one seed): −0.021 on English and −0.018 on Portuguese financial text against the SwiGLU control (`ds-fresh`); both 95% intervals exclude zero. Starting it from a layer-by-layer imitation of the trained block recovers +0.010 on English, but it still finishes below both ordinary controls, and it trains about 2.3× slower per token.
- **Group-rational KAN (GR-KAN)** (three seeds): it first diverged because of the trained input scaling it inherits; with that scaling reset it trains stably. Against a two-matrix SiLU MLP given the same reset, its seed-level intervals include zero on all three domains; on JSON its gain is +0.008 at seed 0 and +0.004 averaged over three seeds. Against SwiGLU it is −0.005 on English and −0.013 on Portuguese financial text, with seed-level intervals that exclude zero.
- **Learnable rational activation** (`mlp-rational-n1`, three seeds; not a KAN): the best of the reset arms. Over the matched SiLU MLP it gains +0.009 on JSON (interval excludes zero) and +0.005 on English (interval includes zero). Against SwiGLU it is +0.003 on English (includes zero) and −0.011 on Portuguese financial text (excludes zero).
- **Training the existing SwiGLU student longer** (`ds-cont`, one seed) gave the highest English score, 0.595. A fresh SwiGLU block scored higher on Portuguese financial text (0.686 against 0.667).

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

The arms form matched comparisons. Each arm differs from the arm in its "Compare with" column by one change; comparisons outside that column (for example `gr-kan-n1` against `ds-fresh`, which changes both the block and the input gains) measure the complete replacement against the SwiGLU baseline:

| Arm | What it is | Compare with |
|---|---|---|
| `ds-ref` | The trained b5 student, the common starting point. Not retrained. | — |
| `ds-cont` | `ds-ref` trained 96M tokens further, all weights kept. Control for the extra training. | `ds-ref` |
| `ds-fresh` | `ds-ref` with its SwiGLU feed-forward block reset to random weights, then trained. Control for discarding the trained block. | `ds-cont` |
| `mlp-silu` | Gate removed: a two-matrix SiLU MLP, widened to 2304 to keep the bytes equal. | `ds-fresh` |
| `kan-bspline` | Per-connection splines (a true KAN). | `mlp-silu`, `ds-fresh` |
| `kan-bspline-fit` | The same KAN, each layer first fitted to imitate the trained D-S feed-forward block. | `kan-bspline` |
| `gr-kan` | Rational function on the inputs of both matrices (KAT layout). Diverged; see section 6.5. | — |
| `mlp-silu-n1` | `mlp-silu` with the pre-block normalization gains reset to 1. | `mlp-silu` |
| `gr-kan-n1` | GR-KAN with the same reset. | `mlp-silu-n1`, `ds-fresh` |
| `mlp-rational-n1` | The SiLU MLP with SiLU replaced by a learnable rational function, same reset. | `mlp-silu-n1`, `ds-fresh` |

The suffix `-n1` means the normalization gains in front of the new block start at 1 instead of the values learned by b5 (section 6.5). The rational MLP without the reset (`mlp-rational`) was not run.

## 5. Setup

- **Starting point.** All arms start from the trained b5 student (96M tokens of distillation; `docs/article.md` section 6.1) and keep its attention, embeddings and output head. `ds-ref` itself is scored as saved and not trained further. Of the trained arms, `ds-cont` keeps the trained block and every other arm starts with a replacement block.
- **Training.** 96M more tokens each, with forward-KL distillation from the target's top-32 probabilities, the b5 data mix, learning rate 6e-4 with cosine decay, 16 sequences of 1024 tokens per step, seed 0 (seeds 0–2 for the arms in section 6.7; the seed sets both the new block's initialization and the data order), `torch.compile`. Equal tokens for every arm.
- **Metric.** Offline greedy agreement: on 100 held-out prompts per domain, the fraction of positions where the student's most likely next token equals the target's. This tracks the acceptance rate of greedy speculative decoding (`docs/article.md` section 6). Each arm is scored on the same prompts, so differences are paired: their 95% confidence intervals come from resampling prompts (2000 draws, positions-weighted) with `configs/analyze.py`.
- **Decision rule (stop rule).** A KAN arm advances only if, on general English, its paired difference against both controls is positive and the lower bound of the 95% interval is above zero. The controls are the fresh SwiGLU block (`ds-fresh`) and the two-matrix SiLU MLP with the same input-gain treatment (`mlp-silu` for the B-spline arms, `mlp-silu-n1` for GR-KAN). Section 6.2 applies the rule to seed 0 with prompt-level intervals; section 6.7 checks it across three training seeds.
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
| `ds-cont` (SwiGLU, trained block kept) | **0.595** [0.576, 0.615] | 0.667 [0.652, 0.681] | 0.883 [0.873, 0.893] |
| `gr-kan-n1` (GR-KAN, input gains reset, section 6.5) | 0.581 [0.562, 0.601] | 0.675 [0.661, 0.690] | 0.881 [0.872, 0.889] |
| `mlp-silu-n1` (two-matrix MLP, input gains reset) | 0.587 [0.567, 0.609] | 0.670 [0.655, 0.685] | 0.873 [0.863, 0.881] |

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
| `kan-bspline-fit` − `ds-cont` (both start from the trained block's function) | −0.021 [−0.028, −0.014] | −0.004 [−0.011, +0.004] | −0.015 [−0.020, −0.010] |
| `ds-fresh` − `ds-cont` (new block against kept block) | −0.010 [−0.016, −0.003] | +0.019 [+0.012, +0.027] | +0.002 [−0.003, +0.006] |
| `ds-cont` − `ds-ref` (96M more tokens) | +0.026 [+0.019, +0.033] | +0.018 [+0.013, +0.024] | −0.005 [−0.009, −0.001] |
| `gr-kan-n1` − `mlp-silu` | −0.000 [−0.006, +0.006] | −0.006 [−0.013, +0.001] | +0.007 [+0.002, +0.012] |
| `gr-kan-n1` − `ds-fresh` | −0.005 [−0.010, +0.001] | −0.011 [−0.017, −0.004] | −0.004 [−0.009, +0.000] |
| `gr-kan-n1` − `mlp-silu-n1` (same reset; the clean GR-KAN test) | −0.006 [−0.013, +0.001] | +0.005 [−0.001, +0.012] | +0.008 [+0.003, +0.013] |
| `mlp-silu-n1` − `mlp-silu` (effect of the reset alone) | +0.006 [+0.001, +0.011] | −0.011 [−0.017, −0.006] | −0.001 [−0.007, +0.004] |

For scale: in each speculative-decoding cycle the draft proposes γ = 4 tokens, and α is the probability that the target accepts each one (offline greedy agreement estimates it). A cycle then yields 1 + α + α² + α³ + α⁴ tokens on average. Raising α from 0.57 to 0.58 raises that from 2.186 to 2.225, about 1.8% more throughput if the cycle time stays fixed (an estimate from the speed model in `docs/article.md` section 3.3; CPU draft speed was not measured here). The KAN's loss of 0.02 is small in absolute terms but consistent, and it comes on top of the slower training.

### 6.3 How agreement develops

General-English agreement during training (measured on the same prompts every 4M tokens):

| Tokens | `ds-fresh` | `mlp-silu` | `kan-bspline` | `kan-bspline-fit` |
|---|---|---|---|---|
| 4M | 0.389 | 0.444 | 0.382 | **0.505** |
| 20M | 0.514 | 0.518 | 0.496 | **0.526** |
| 52M | **0.562** | 0.555 | 0.543 | 0.554 |
| 92M | 0.585 | **0.588** | 0.566 | 0.577 |

The fitted KAN starts far ahead, since it already imitates the trained block: +0.12 over `ds-fresh` at 4M tokens. The lead shrinks to about 0.005 by 24M tokens, and in the full curve (every 4M tokens) `ds-fresh` passes it at 52M. The plain KAN trails both ordinary blocks at every checkpoint shown; at 96M its paired English gaps are 0.021 to `ds-fresh` and 0.017 to `mlp-silu`.

### 6.4 How well a KAN can imitate the trained block

Before training, each `kan-bspline-fit` layer was fitted for 1954 steps (two passes over 4M tokens of teacher data) to reproduce the trained SwiGLU block's output on the same inputs. The remaining error, as a fraction of the output's variance, measured on batches not used for fitting:

| Layers | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Relative error after fit | 0.25 | 0.20 | 0.23 | 0.31 | 0.38 | 0.39 | 0.33 | 0.33 | 0.37 | 0.36 | 0.45 | 0.22 |

So a byte-matched KAN, 256 units wide, captures about 55–80% of what the 1536-unit SwiGLU block computes. That is the head start seen in section 6.3, and also its limit: the remaining error compounds through the layers, and further training does not close the gap.

### 6.5 The rational variants

**First attempt: divergence.** The GR-KAN arm (`gr-kan`) failed from the first step: loss 10.4, gradient norm 2.7 × 10⁷, and NaN by step 11. The cause is the input it receives. Each block's input passes through the b5 student's normalization layer, whose trained per-channel gains reach 191 in the last layer, so the new rational unit saw inputs up to 92. A degree-5 polynomial evaluated at 92 is about 6.6 × 10⁹, and its coefficients' gradients swamped everything else.

A second weakness is built into the rational form itself. The SiLU-shaped starting function is accurate on the interval it was fitted on (±4, error 0.014), but far outside it, it behaves like a straight line on *both* sides: f(−30) = −3.4 and f(100) = 661, where SiLU gives about 0 and 100. A ratio of a degree-5 and a degree-4 polynomial cannot follow SiLU's flat left tail. The original GR-KAN avoids both problems because its inputs come straight out of a normalization layer with gains near 1.

**Second attempt: inputs reset.** In `gr-kan-n1`, the normalization gains in front of the new block are reset to 1 when the block is created. This discards the input scaling learned by b5; the cost of that is measured separately below, by training the same SiLU MLP once with inherited gains and once with gains reset to 1. On a CPU probe, this cut the first-step gradient norm from 7.9 × 10⁶ to 335. The run trained stably: it started slowest of all arms (0.286 on English at 4M tokens, against 0.38–0.51 for the others), caught up by about 20M tokens, and finished level with the two-matrix MLP trained without the reset (−0.000 on English, −0.006 on Portuguese financial text, +0.007 on JSON). It stays below the SwiGLU block on Portuguese financial text (−0.011, interval excludes zero); on the other two domains it is also below, but those intervals include zero.

**The clean comparison.** The reset itself changes results: the two-matrix MLP with reset gains (`mlp-silu-n1`) gains 0.006 on English and loses 0.011 on Portuguese financial text against the same MLP without the reset. GR-KAN must therefore be compared with `mlp-silu-n1`, which received the same reset. Against it, GR-KAN is −0.006 on English and +0.005 on Portuguese financial text (both intervals include zero) and +0.008 on JSON (excludes zero). Learnable rational activations help slightly on structured output only, and they do not reach the SwiGLU block on the prose domains. These are seed-0 numbers; with three seeds the JSON gain is +0.004 and no longer resolved (section 6.7).

**The rational MLP.** The learnable-rational activation alone, on the two-matrix MLP and with the same gain reset (`mlp-rational-n1`), was run on 2026-11-27 with three seeds. It trained stably from the first step. It is the strongest of the gains-reset arms, and GR-KAN is below it on English (−0.008) and JSON (−0.005), both with seed-level intervals that exclude zero: placing the rational unit on the inputs of both matrices, as GR-KAN does, is worse than using it once as the activation. The variant without the reset (`mlp-rational`) was not run.

### 6.6 Training cost

| Arm | Wall time for 96M tokens | Steady speed (tokens/s) |
|---|---|---|
| `ds-fresh` (SwiGLU) | 50 min | about 32,000 |
| `mlp-silu` | 65 min | about 28,000 |
| `kan-bspline` | 1 h 57 min | about 14,200 |
| `kan-bspline-fit` | 1 h 58 min, including a 17-minute fit | about 14,200 |
| `gr-kan-n1` | 1 h 48 min | about 15,000 |

The seed runs on 2026-11-27 ran one at a time on an otherwise quieter machine and are faster across the board. Median steady speed, same day, seeds 1–2:

| Arm | Wall time for 96M tokens | Steady speed (tokens/s) |
|---|---|---|
| `ds-fresh` (SwiGLU) | 38–43 min | about 49,000 |
| `mlp-silu-n1` | 37–40 min | about 47,000 |
| `mlp-rational-n1` | 65–69 min | about 26,700 |
| `gr-kan-n1` | 73–78 min | about 23,700 |

`mlp-rational-n1` seed 0 took 2 h 7 min because an unrelated GPU job ran during its first half hour; its steady speed matches the other seeds. CPU scoring overlapped with part of `gr-kan-n1` seed 2 and slowed it by about 20% for 17 minutes. The rational blocks train at 0.48–0.57× the speed of the SwiGLU and SiLU blocks.

Some runs on 2026-11-26 shared the machine with analysis jobs, which slowed the SwiGLU control `ds-cont` to 1 h 26 min, so differences of 10–20% between days are not meaningful. The factor of about 2 between the KAN-type blocks and the ordinary ones is.

Without activation checkpointing (recomputing the block in the backward pass instead of storing its intermediate values), the rational and KAN blocks did not fit in 8 GB even at one sequence per step. With it, their peak memory matches SwiGLU, and `torch.compile` raises the KAN from 4,700 to 14,000–21,000 tokens/s. On the CPU, where the draft runs, the KAN's basis evaluation is extra work per token whose cost is unmeasured.

### 6.7 Three seeds

Seeds 1 and 2 were added on 2026-11-27 for `ds-fresh`, `mlp-silu-n1`, `gr-kan-n1` and the new `mlp-rational-n1` (seeds 0–2). Every model was rescored on the CPU in float32; for the arms shared with section 6.1, the seed-0 scores match it (identical to four decimals in the results files). Results: `experiments/stage-b-distillation/b10-kan-ffn/results/b10-20261127/analysis-cpu-seeds.json`.

Agreement per seed, general English:

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| `ds-fresh` (SwiGLU, new block) | 0.585 | 0.590 | 0.586 | 0.587 |
| `mlp-silu-n1` (two-matrix MLP, gains reset) | 0.587 | 0.582 | 0.586 | 0.585 |
| `mlp-rational-n1` (rational activation, gains reset) | 0.587 | 0.592 | 0.591 | **0.590** |
| `gr-kan-n1` (GR-KAN, gains reset) | 0.581 | 0.586 | 0.579 | 0.582 |

The same seed moves an arm by up to 0.007 on English, as much as most of the single-seed differences in section 6.2.

Paired differences, mean over the three seeds. Each cell reads `mean [prompt 95% CI] {seed 95% CI}`:

- **[prompt CI]** keeps the trained models fixed and resamples the 100 prompts (as in section 6.2), using the per-prompt difference summed over the three seed pairs. It shows how much the result depends on which prompts were chosen, not on training luck.
- **{seed CI}** keeps the prompts fixed and treats the three per-seed differences as the sample (t distribution, 2 degrees of freedom). With the prompts fixed, it estimates the mean paired difference across training seeds; individual runs can still differ in direction. This is the one that matters for a claim about the method, and with three seeds it is wide.

Example: GR-KAN − SiLU MLP on JSON is +0.004 [+0.002, +0.007] {−0.004, +0.013}. On these trained models the gain is consistent across prompts, but across training seeds its direction is unresolved.

| Comparison | General English | Portuguese financial | JSON |
|---|---|---|---|
| `gr-kan-n1` − `mlp-silu-n1` | −0.003 [−0.006, +0.001] {−0.017, +0.011} | +0.002 [−0.003, +0.006] {−0.007, +0.010} | +0.004 [+0.002, +0.007] {−0.004, +0.013} |
| `mlp-rational-n1` − `mlp-silu-n1` | +0.005 [+0.001, +0.009] {−0.007, +0.017} | +0.004 [+0.000, +0.009] {−0.002, +0.010} | **+0.009** [+0.006, +0.012] {+0.003, +0.015} |
| `gr-kan-n1` − `mlp-rational-n1` | **−0.008** [−0.011, −0.005] {−0.016, −0.0002} | −0.003 [−0.006, +0.001] {−0.010, +0.005} | **−0.005** [−0.008, −0.002] {−0.007, −0.003} |
| `gr-kan-n1` − `ds-fresh` | **−0.005** [−0.009, −0.002] {−0.009, −0.001} | **−0.013** [−0.017, −0.009] {−0.019, −0.008} | −0.000 [−0.003, +0.003] {−0.009, +0.009} |
| `mlp-rational-n1` − `ds-fresh` | +0.003 [−0.000, +0.006] {−0.001, +0.007} | **−0.011** [−0.015, −0.006] {−0.016, −0.005} | +0.005 [+0.002, +0.007] {−0.006, +0.015} |

Bold: both intervals exclude zero. Per-seed differences and their standard deviation are in the results file.

- **GR-KAN fails the stop rule with three seeds as with one.** Its seed-level intervals against its matched MLP include zero on every domain, and it is below the SwiGLU block on both prose domains.
- **The learnable rational activation is a small, real improvement over SiLU only on JSON** (+0.009). Its English gain (+0.005) has a seed-level interval that includes zero.
- **No variant beats the SwiGLU block the student already uses.** The rational MLP is level with it on English and JSON and loses 0.011 on Portuguese financial text.
- **Single-seed conclusions were too confident in one place.** The +0.008 JSON gain of GR-KAN over its matched MLP (section 6.5) is +0.004 on three seeds, and its seed-level interval includes zero.

## 7. Reading the results

- **Every KAN arm fails the experiment's stop rule.** The rule (section 5) asks for a positive English difference against both controls with the lower interval bound above zero. Both B-spline arms are below both controls; GR-KAN does not differ significantly from its matched MLP on English and is below SwiGLU.
- **Imitating the trained block is a good start, not a good model.** The fitted start gives +0.12 at 4M tokens and +0.010 at the end on English, but it is worse on JSON (−0.011), and it does not reach the ordinary blocks.
- **The result agrees with the independent comparison** of Yu et al. (2024): at matched parameters, KANs lose to ordinary layers outside symbolic-formula fitting. Here the comparison is matched in runtime bytes, which is what matters for a cache-resident draft.
- **Removing the gate costs a little on JSON.** In seed 0, replacing the 1536-wide SwiGLU block with the byte-matched 2304-wide SiLU MLP lowers JSON agreement by 0.011 [−0.017, −0.006]; both prose differences are −0.005 with intervals that include zero. The gated SwiGLU block stays the baseline for the student.
- **Training a fresh block for 96M more tokens helps** (`ds-fresh` against the b5 start: +0.016 on English, +0.038 on Portuguese financial text). Keeping the trained block helps more on English (`ds-cont`: +0.026 over the start, and +0.010 over `ds-fresh`), while the fresh block learns the Portuguese financial domain better (+0.019 over `ds-cont`). This is the largest effect measured tonight, and it has nothing to do with KANs: the student was not yet saturated at 96M tokens.

## 8. Limitations

- **Seeds.** The B-spline arms, `mlp-silu` and `ds-cont` have one seed; their intervals cover prompt variation only. `ds-fresh`, `mlp-silu-n1`, `mlp-rational-n1` and `gr-kan-n1` have three seeds (section 6.7). With three seeds, precision varies by comparison: on English, the rational MLP's +0.005 over the SiLU MLP has a seed interval of {−0.007, +0.017}, while GR-KAN's −0.005 against `ds-fresh` has {−0.009, −0.001}.
- **Development evaluation.** The 100 prompts per domain were used throughout development by the same agents that ran and scored the experiment. They are not a sealed test set (see the status note).
- **Agreement, not speed.** No KAN draft ran inside llama.cpp, so there is no end-to-end speed. Because the KAN loses on agreement, the speed test was not built.
- **Inherited input scaling.** Both B-spline arms inherited b5's trained input gains. The reset was tested for GR-KAN, the SiLU MLP and the rational MLP, but never for a spline arm. The spline arms' inputs pass through those inherited gains, which reach 191 in the last layer; the `-n1` arms start with them set to 1. In the trained KAN, 0.5–8.5% of the spline inputs fall outside the spline grid [−2, 2] in layers 0–10, and 37% in layer 11; there, only the block's ordinary weight path is active. A grid sized to the inputs might narrow the gap.
- **Untuned KAN settings.** One grid size (G = 5), one grid range, and one fitting schedule were tried.
- **Scoring precision.** These scores were computed in float32 on the CPU. The training-time scores, in bfloat16 on the GPU, differ by about 0.001 (0.5693 against 0.5679 for the b5 student on English).

## 9. What next

- **Close the KAN deployment work.** Neither the B-spline nor the GR-KAN block cleared the decision rule, so the existing SwiGLU CPU graph stays, and neither the spline basis kernel nor the rational-function kernel for llama.cpp is built.
- **Do not port the learnable rational activation either.** Against `mlp-silu-n1`, only its JSON gain (+0.009) has a seed interval above zero; against `ds-fresh`, its seed-level intervals include zero on English and JSON and lie below zero on Portuguese financial text; and it trains about 1.8× slower. Reopen only if its English seed-level interval against both controls has a positive lower bound.
- **Train the SwiGLU student longer.** `ds-cont` gained +0.026 on English from 96M more tokens, and agreement is what the speed depends on (`docs/article.md` section 3.3). Proposed setup:
  - resume the final `ds-cont` checkpoint (192M cumulative tokens: 96M in b5 plus 96M here) for another 96M tokens, with seeds 0–2 for data order;
  - restart the learning-rate schedule (6e-4, 100 warmup steps, cosine decay) and keep the batch, data mix and distillation loss of section 5;
  - score the start and end checkpoints on the same prompts in CPU float32;
  - count it as an improvement if the English seed-level interval of end − start has a lower bound above zero, and report the other two domains alongside.

  The launch script, data mix and teacher-target paths are in `experiments/stage-b-distillation/b10-kan-ffn/configs/run_b10.sh` (arm `ds-cont`).
- **If KANs are revisited**, a fair retest would differ from this one in the input handling only. Proposed setup:
  - run `ds-fresh`, the SiLU MLP, the rational MLP, GR-KAN and both spline arms on seeds 0–2, all with the pre-block gains reset to 1;
  - for both spline arms, also run a per-layer symmetric grid whose radius is the 99th percentile of the absolute spline inputs on a fixed 4M-token sample, against the current [−2, 2] grid;
  - keep G = 5, cubic splines, the widths and the 96M-token budget;
  - report the fraction of inputs inside the grid and all three domains;
  - advance a KAN only under the section 5 rule, using seed-level intervals against the reset controls.

## References

- Liu et al. 2024, KAN: Kolmogorov-Arnold Networks. https://arxiv.org/abs/2404.19756
- Yang and Wang, Kolmogorov-Arnold Transformer (ICLR 2025). https://arxiv.org/abs/2409.10594
- Yu, Yu and Wang 2024, KAN or MLP: A Fairer Comparison. https://arxiv.org/abs/2407.16674
- Li 2024, Kolmogorov-Arnold Networks are Radial Basis Function Networks (FastKAN). https://arxiv.org/abs/2405.06721
