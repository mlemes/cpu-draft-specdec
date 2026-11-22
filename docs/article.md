# A small CPU model that drafts for a large GPU model: speculative decoding and distillation on a desktop PC

*Status: working article, updated 2026-11-22. It covers the feasibility study, the first distillation experiments, the first end-to-end speed measurements of the distilled draft, and longer training runs.*

## 1. The problem

A large language model writes text one token at a time. A token is a word or a piece of a word. To produce each token the model reads all of its weights once. On a consumer graphics card the bottleneck is memory bandwidth, not arithmetic: an 8-billion-parameter model compressed to about 4.5 bits per weight occupies roughly 5 GB, and moving those bytes through the GPU for every token takes about 12–13 ms on an RTX 3070. The GPU's compute units sit mostly idle during that time.

Speculative decoding exploits this idle compute. A small, fast model guesses several tokens ahead, and the large model checks all of the guesses in a single pass. Checking five tokens costs almost the same as generating one, because the weights are read once either way. When the guesses are good, several tokens come out of each expensive pass.

This project asks a hardware question on top of that idea. A desktop PC also has a CPU, which is usually idle during GPU inference. The CPU studied here, an AMD Ryzen 5 5500X3D, has an unusually large 96 MB last-level (L3) cache. A draft model whose working weights fit in that cache could run from on-chip memory at high speed without touching main memory. Can such a CPU-resident draft speed up the GPU-resident model? And can a draft small enough to fit be trained to guess well?

## 2. Setup

| Component | Choice |
|---|---|
| Large model (the *target*) | Qwen3-8B, 4-bit quantized (Q4_K_M), on an RTX 3070 with 8 GB |
| Small model (the *draft*) | Trained in this project; about 50 million parameters in its compute path, on the CPU |
| CPU | Ryzen 5 5500X3D, 6 cores, 96 MB L3 cache |
| Inference engine | llama.cpp |
| Training | PyTorch on the same RTX 3070 |
| Evaluation domains | General English text (Wikipedia), Portuguese financial text (minutes of the Brazilian central bank's monetary policy committee), and JSON output for structured requests |

All data is public. The financial minutes used for evaluation are later documents than the ones used for training, so the model never sees the evaluation text.

## 3. Speculative decoding

### 3.1 The procedure

Let $p$ be the target's next-token distribution and $q$ the draft's. One cycle of speculative decoding with block length $\gamma$ works as follows:

1. The draft generates $\gamma$ tokens $x_1, \dots, x_\gamma$ one at a time, sampling each from $q(\cdot \mid \text{context}, x_{<i})$.
2. The target scores the context plus all $\gamma$ drafted tokens in one forward pass. This yields $p(\cdot \mid \text{context}, x_{<i})$ for every position $i = 1, \dots, \gamma + 1$.
3. Drafted tokens are accepted left to right. Token $x_i$ is accepted with probability

$$
\min\!\left(1, \frac{p(x_i)}{q(x_i)}\right).
$$

4. At the first rejection, a replacement token is sampled from the residual distribution

$$
p'(x) = \frac{\max\big(0,\, p(x) - q(x)\big)}{\sum_y \max\big(0,\, p(y) - q(y)\big)},
$$

   and the cycle ends. If all $\gamma$ tokens are accepted, the target's distribution at position $\gamma + 1$ is already available, so one extra token is sampled from it for free.

This rule makes the output distribution exactly equal to the target's. Speculative decoding changes speed, never the text distribution. In greedy decoding (always taking the most likely token) the rule reduces to a simple check: $x_i$ is accepted if and only if it equals the target's most likely token at that position.

### 3.2 How many tokens per cycle

Let $\alpha$ be the probability that a single drafted token is accepted. For a given position, the expected acceptance probability under the rule above is

$$
\alpha = \sum_x \min\big(p(x), q(x)\big) = 1 - \mathrm{TV}(p, q),
$$

where $\mathrm{TV}$ is the total-variation distance between the two distributions. In greedy decoding, $\alpha$ is the probability that the draft's top token matches the target's top token.

If acceptances are treated as independent with a constant $\alpha$, the number of tokens produced per cycle (accepted drafts plus the one token the target always contributes) has expectation

$$
E(\alpha, \gamma) = \sum_{i=0}^{\gamma} \alpha^i = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}.
$$

With $\gamma = 4$, a draft that agrees half the time ($\alpha = 0.5$) yields $E \approx 1.94$ tokens per cycle; $\alpha = 0.8$ yields $E \approx 3.36$.

### 3.3 When it pays off

Let $t_t$ be the target's time to generate one token normally, $t_d$ the draft's time per token, $t_v$ the target's time to verify a block of $\gamma + 1$ tokens, and $t_o$ any fixed per-cycle overhead (synchronization, sampling, bookkeeping). The speedup over plain target decoding is

$$
S = \frac{E(\alpha, \gamma)\; t_t}{\gamma\, t_d + t_v + t_o}.
$$

The numerator is the time plain decoding would need for the tokens one cycle produces; the denominator is what the cycle actually costs. Speculative decoding wins when $S > 1$. Solving $S = 1$ for $\alpha$ gives the *break-even acceptance rate*, the minimum agreement a draft needs to be worth running.

On this machine the measured values were $t_t \approx 12.6$ ms and $t_v \approx 14.2$ ms for a block of five tokens ($\gamma = 4$). Verifying five tokens costs only 12% more than generating one, which is why the method works. For a draft of the planned size on the CPU ($t_d \approx 1.1$ ms) and the roughly 2–3 ms of loop overhead measured in llama.cpp, the model gives:

| Goal | Required $\alpha$ at $\gamma = 4$ |
|---|---|
| Break even ($S = 1.0$) | about 0.40 |
| 1.3× speedup | about 0.55 |

These two thresholds are the targets for the trained draft.

### 3.4 Why the CPU and why the cache matters

A draft that fits in the L3 cache does not need main-memory bandwidth, which the CPU has far less of than the GPU. Two measurements anchor this:

- A model-sized block of about 56 MB of weights decoded at 1.14 ms per token on three CPU cores, with only 3–5% of the touched bytes coming from main memory.
- The same computation, with the cache allocation artificially restricted to 6 MB through the CPU's cache-partitioning feature, took 2.41 ms per token. Cache residency alone is worth a factor of 2.5.

The size limit is strict. An off-the-shelf 0.6-billion-parameter model (Qwen3-0.6B, about 600 MB at 8 bits) does not fit, and on the CPU it *slowed* generation to 0.25–0.47× of plain GPU decoding. The draft must be small, and small models guess poorly unless they are trained for the job.

## 4. Feasibility study

Before training anything, a set of measurements checked whether the idea could work at all.

| Question | Finding |
|---|---|
| Is the target's verify pass cheap? | Yes: 14.2 ms to check five tokens vs 12.6 ms to generate one. |
| Can a draft of the planned size run from the cache? | Yes: about 1.1 ms per token on three cores, with little main-memory traffic. |
| Can the draft's output vocabulary be cut? | Yes. Qwen3 has 151,936 tokens; restricting the draft to the 32,768 the target uses most still covers 96–98% of the target's probability mass, depending on the domain. |
| How well does an untrained small model agree? | Qwen3-0.6B, from the same family as the target, reaches $\alpha \approx 0.63$–$0.66$ on prose and 0.99 on JSON. It is too large for the cache, though. |
| Does a cache-sized CPU draft speed anything up in practice? | Yes, with a different model pair. A public 68-million-parameter draft for Llama-2-7B gave 1.40× on English and 1.12× on Portuguese financial text, running on the CPU. |
| Is the CPU better than the GPU for the draft? | No, not in llama.cpp's current loop. The GPU runs the same 68M draft at 0.26 ms per token against 0.63 ms on the CPU, and the GPU version gave 1.65× and 1.32×. |

The last row reshaped the project. llama.cpp's speculative loop is strictly sequential: draft, then verify, then draft again. When the draft runs on the GPU it is simply faster, so a CPU draft cannot win on speed in that loop. A careful breakdown of one cycle confirmed that the gap comes entirely from the draft's own compute time, with no hidden fixed cost.

A CPU draft can still pay off in two ways:

1. **Overlap.** The CPU drafts the next block while the GPU verifies the current one. The estimated ceiling for this on the measured timings is a cycle of about 13.2 ms instead of 16.3 ms, roughly 7% faster than a GPU draft. That is real but modest, and requires changing the engine.
2. **Freeing the GPU.** A draft on the CPU uses no GPU memory and no GPU time. On an 8 GB card that matters: the memory can hold a longer context, a larger target, or other work.

The project therefore continued with distillation first and treats the CPU placement mainly as a way to keep the GPU free.

## 5. Distillation

### 5.1 The idea

*Knowledge distillation* trains a small *student* model to imitate a large *teacher*. Ordinary language-model training shows the model a text and asks it to predict each next token; the training signal is a single correct answer per position (a *hard label*). In distillation the teacher first reads the same text, and the student learns to match the teacher's full probability distribution at every position. A distribution such as "the 70%, a 20%, this 5%, …" carries much more information than the single word that happened to appear, including which alternatives are plausible and how uncertain the teacher is.

For speculative decoding this aligns directly with the goal. The acceptance rate is $1 - \mathrm{TV}(p, q)$, so a student whose distribution $q$ is close to the teacher's $p$ is, by construction, a good draft.

### 5.2 The loss

Storing the teacher's full distribution over 151,936 tokens for every training position would take far too much disk. Instead, the teacher's 32 most likely tokens and their probabilities are stored, and everything else is lumped into one *tail* bucket. Let $K$ be the set of stored tokens, $p_k$ their teacher probabilities, $p_\text{rest} = 1 - \sum_{k \in K} p_k$, and correspondingly $q_k$ and $q_\text{rest}$ for the student. The training loss per position is the forward Kullback–Leibler divergence over this coarsened distribution:

$$
\mathcal{L}_\text{KD} = \sum_{k \in K} p_k \log \frac{p_k}{q_k} \;+\; p_\text{rest} \log \frac{p_\text{rest}}{q_\text{rest}}.
$$

The top 32 tokens hold about 96% of the teacher's probability on prose and 99.5% on JSON, so little information is lost. The baseline loss for comparison is ordinary cross-entropy against the next token in the text, $\mathcal{L}_\text{hard} = -\log q(x_\text{next})$.

### 5.3 The student

The student is a dense transformer: every parameter takes part in every token. It uses the same architecture family and the same tokenizer as Qwen3, so its predictions are directly comparable with the teacher's.

| | Student | Qwen3-0.6B (for scale) |
|---|---|---|
| Layers | 12 | 28 |
| Hidden width | 512 | 1,024 |
| Attention heads (query / key-value) | 4 / 1 | 16 / 8 |
| Feed-forward width | 1,536 | 3,072 |
| Output vocabulary | 32,768 most-used tokens | 151,936 |
| Parameters in the compute path | about 53 million | about 440 million |

The reduced output vocabulary matters because the output layer is otherwise the largest single matrix the draft reads per token. The single key-value head keeps the per-token memory of the attention cache small.

### 5.4 Starting point: pruning

A student can start from random weights or from a shrunk copy of an existing model. *Pruning* here means building the student's initial weights by selecting parts of Qwen3-0.6B: 12 of its 28 layers spread evenly through its depth, the most important half of its hidden dimensions, its most important feed-forward neurons, and averaged attention heads, plus the output rows for the 32,768 kept tokens. The pruned model is poor at first, but it starts with useful structure, such as word embeddings and attention patterns, that a random model must learn from scratch.

### 5.5 The data pipeline

1. **Training text** (about 25 million tokens): Wikipedia in English and Portuguese, older central-bank minutes, and examples of structured JSON answers.
2. **Teacher scoring:** the 8B model reads all training text on the GPU and stores its top 32 probabilities per position (about 2,240 tokens per second, three hours in total, 4.5 GB on disk).
3. **Student training:** the student reads the same text and minimizes the loss above, with the domains mixed in fixed proportions.
4. **Offline evaluation:** the 8B model generates greedy continuations of held-out prompts (100 per domain, up to 128 tokens each). For each generated position, the measure is whether the student's top token equals the target's. That fraction is the greedy acceptance rate $\alpha$, measured without running the full speculative loop. Confidence intervals come from resampling prompts.

## 6. First results

Three students were trained with the same data, size, and budget of 24 million training tokens. Each run took about 14 minutes on the RTX 3070. They differ in how they learn and where they start, which isolates the effect of each choice.

| Student | Learns from | Starts from | English | Portuguese financial | JSON |
|---|---|---|---|---|---|
| Distilled, pruned start | Teacher's probabilities | Shrunk Qwen3-0.6B | **0.44** (0.42–0.46) | **0.59** (0.58–0.60) | **0.53** (0.51–0.55) |
| Hard labels, pruned start | Next token in the text | Shrunk Qwen3-0.6B | 0.40 (0.39–0.42) | 0.48 (0.46–0.49) | 0.47 (0.45–0.49) |
| Distilled, random start | Teacher's probabilities | Random weights | 0.41 (0.39–0.43) | 0.58 (0.56–0.59) | 0.43 (0.40–0.45) |
| *Untrained Qwen3-0.6B (8× larger)* | | | *0.63* | *0.66* | *0.99* |
| *Needed to break even* | | | *≈ 0.40* | *≈ 0.40* | *≈ 0.40* |
| *Needed for 1.3×* | | | *≈ 0.55* | *≈ 0.55* | *≈ 0.55* |

Values are greedy acceptance rates with 95% confidence intervals.

**Distillation helps.** Learning from the teacher's probabilities beats learning from the text alone in every domain, by 0.04 on English, 0.06 on JSON, and 0.11 on Portuguese financial text.

**The pruned start helps.** Starting from a shrunk Qwen3-0.6B beats random initialization by 0.03 on English and 0.10 on JSON. On the financial text the two are statistically indistinguishable.

**The targets are partly met.** The best student clears the break-even rate in all three domains and the 1.3× rate on the financial text. Plugging its rates into the cost model of section 3.3 ($\gamma = 4$, $t_d = 1.1$ ms, $t_o = 3$ ms) gives estimated speedups of about 1.03× on English and 1.32× on the financial text. Section 7 compares these estimates with measurements.

**Training was not finished.** Over the last quarter of the run, agreement still rose by 0.01–0.03 for every 4 million tokens, even with the learning rate decayed to 10% of its peak.

**A caveat on the financial domain.** Its training text is small (about 136,000 tokens), so the student saw it roughly nine times. The evaluation minutes are held out, so this is not memorization of the test text, but part of the high score may reflect the minutes' formulaic style.

### 6.1 Longer training and more structured data

Two follow-up runs kept the best recipe (distilled, pruned start) and changed one thing at a time. The first trained four times longer, on 96 million tokens instead of 24 million, about 1 hour on the RTX 3070. Because the scored training text holds only about 25 million tokens, the student saw it about four times. The second run kept the 96 million tokens but raised the share of structured output in the training mix from 8% to 23%, adding about 9,500 function-calling and JSON-mode conversations from the same public collection. None of these overlap the JSON evaluation prompts.

| Student | English | Portuguese financial | JSON |
|---|---|---|---|
| 24 million tokens (section 6) | 0.44 (0.42–0.46) | 0.59 (0.58–0.60) | 0.53 (0.51–0.55) |
| 96 million tokens | 0.56 (0.54–0.58) | 0.65 (0.64–0.67) | 0.75 (0.73–0.77) |
| 96 million tokens, 23% structured output | 0.56 (0.54–0.58) | 0.65 (0.64–0.66) | **0.89 (0.88–0.90)** |
| *Untrained Qwen3-0.6B (8× larger)* | *0.63* | *0.66* | *0.99* |

**Longer training pays off, then saturates.** Four times the training raised agreement by 0.13 on English, 0.06 on the financial text and 0.22 on JSON. The financial text now matches the 0.6B model, and English crosses the rate needed for 1.3×. Most of the gain came in the first 60 million tokens. After that, agreement on prose barely moved, which is the expected sign of a student repeating the same text: further gains on prose need new text scored by the teacher, not more passes over the old.

**More structured data helps structured output at no cost to prose.** The larger share raised JSON agreement from 0.75 to 0.89, still rising at the end of the run, while English and financial text stayed within their confidence intervals. The structured examples replaced repeated passes over prose text that the student had already saturated, so prose lost nothing.

## 7. Measured speedups

### 7.1 Running the student inside llama.cpp

llama.cpp expects a draft model to produce a score for every token of the target's vocabulary. The student scores only its 32,768 kept tokens, so a small change to llama.cpp was needed. The model file now carries a map from each vocabulary token to its row in the student's output layer. After computing the reduced scores, the engine expands them to the full vocabulary and gives every token outside the kept set a score of minus infinity, so it can never be drafted. The rest of the engine, including sampling and the speculative loop, is unchanged.

Two checks confirmed the conversion. The engine's probabilities for the converted student match those of the original training code. And speculative decoding with the student as the draft produces exactly the same text as the target alone, as the method guarantees. The converted student occupies 139 MB at 8-bit precision. About 82 MB of that is the input embedding table, of which only one row is read per token.

### 7.2 Results

The best student from section 6 was run as the draft for Qwen3-8B in llama.cpp's speculative loop. The prompts and settings were the same as in the earlier measurement with Qwen3-0.6B as the draft: greedy decoding, 64 generated tokens, 100 prompts per domain for $\gamma = 4$ and 30 for $\gamma = 2$ and 6. The baseline is the target alone, at 76 tokens per second.

| Block length $\gamma$ | Draft runs on | English | Portuguese financial | JSON |
|---|---|---|---|---|
| 2 | CPU | 1.04 (1.00–1.07) | **1.27 (1.23–1.31)** | 1.07 (1.03–1.11) |
| 2 | GPU | 1.05 (1.01–1.09) | 1.26 (1.23–1.30) | 1.11 (1.07–1.15) |
| 4 | CPU | 0.92 (0.89–0.94) | 1.18 (1.15–1.21) | 1.02 (0.99–1.05) |
| 4 | GPU | 0.96 (0.93–0.98) | 1.26 (1.23–1.29) | 1.09 (1.06–1.12) |
| 4 | *GPU, Qwen3-0.6B as draft* | *1.03* | *1.09* | *1.90* |

Values are speedups over the target alone, with 95% confidence intervals. Longer blocks ($\gamma = 6$) were slower in every domain.

**The offline measure predicts the real one.** The engine reports how many drafted tokens it accepted. Converting that count back into a per-position acceptance rate with the formula of section 3.2 gives 0.43 on English, 0.60 on the financial text and 0.51 on JSON. The offline evaluation of section 6 predicted 0.44, 0.59 and 0.53. Students can therefore be compared offline, in minutes, without full speed runs.

**At short blocks the CPU draft costs almost nothing.** At $\gamma = 2$ the CPU and GPU versions of this draft are statistically indistinguishable in every domain. The GPU pulls ahead only at longer blocks, where the CPU's slower drafting accumulates over more tokens. Since short blocks are also the fastest setting at this level of agreement, the CPU draft delivers the full speedup while using no GPU memory and no GPU time.

**A distilled 50M draft beats an off-the-shelf model eight times its size on prose.** On the financial text the student reaches 1.27× against 1.09× for Qwen3-0.6B, and on English 1.05× against 1.03×. On JSON it falls far behind (1.11× against 1.90×). The 0.6B model agrees with the target almost perfectly on structured output, while the student, which saw JSON in only 8% of its training data, does not.

**English sits at break-even,** as the cost model predicted for an acceptance rate of about 0.44.

### 7.3 The longer-trained draft

The student trained on 96 million tokens (section 6.1, with JSON at 8% of the mix) was measured the same way, adding $\gamma = 3$. The baseline is again the target alone, at 77.5 tokens per second.

| Block length $\gamma$ | Draft runs on | English | Portuguese financial | JSON |
|---|---|---|---|---|
| 2 | CPU | **1.18 (1.16–1.21)** | 1.31 (1.29–1.33) | 1.43 (1.40–1.46) |
| 2 | GPU | 1.21 (1.18–1.24) | 1.36 (1.33–1.38) | 1.48 (1.45–1.51) |
| 3 | CPU | 1.12 (1.08–1.16) | **1.32 (1.27–1.37)** | 1.47 (1.41–1.53) |
| 3 | GPU | 1.15 (1.11–1.19) | 1.37 (1.32–1.42) | 1.52 (1.45–1.58) |
| 4 | CPU | 1.06 (1.03–1.09) | 1.28 (1.25–1.31) | **1.50 (1.46–1.55)** |
| 4 | GPU | 1.11 (1.08–1.14) | 1.34 (1.31–1.38) | 1.59 (1.53–1.64) |
| 4 | *GPU, Qwen3-0.6B as draft* | *1.03* | *1.09* | *1.90* |

Values are greedy speedups over the target alone with 95% confidence intervals; $\gamma = 3$ used 30 prompts per domain, the others 100. At $\gamma = 6$ every setting was slower than at $\gamma = 4$.

**Longer training turns into speed.** With the draft on the CPU, the best speedups rose from 1.04× to 1.18× on English, from 1.27× to 1.32× on the financial text and from 1.07× to 1.50× on JSON. On prose the 50-million-parameter CPU draft now beats the 0.6B GPU draft by 0.15–0.23.

**The best block length grows with agreement:** 2 tokens on English, 3 on the financial text and 4 on JSON, as the cost model of section 3.3 predicts. A deployed system would choose $\gamma$ per request type, or adapt it on the fly.

**The offline measure stays reliable.** The acceptance rates implied by the engine's counts are 0.53, 0.64 and 0.72, against offline predictions of 0.56, 0.65 and 0.75: slightly optimistic, but in the same order.

**The CPU now trails the GPU slightly.** With higher agreement each cycle carries more accepted tokens, so drafting is a larger share of the cycle and the CPU's slower drafting shows: the CPU version is 0.03–0.10 behind the GPU version of the same draft. The trade remains a few percent of speed for keeping the draft entirely off the graphics card.

## 8. What remains

1. **Measure the structured-output student as a draft.** Its offline JSON agreement of 0.89 should lift the 1.50× of section 7.3 further.
2. **More distinct prose.** Prose agreement saturates once the student repeats its text, so the next run adds about 40 million newly scored tokens of English and Portuguese Wikipedia.
3. **Quantify the GPU memory saved.** The CPU draft matches the GPU draft's speed at short blocks, so the memory it leaves free on the card is its main benefit and should be measured as such.
4. **Overlap drafting and verification.** A modified loop that drafts on the CPU while the GPU verifies could add up to about 7% over a GPU draft.
5. **Conditional computation.** The original plan also compares the dense student with a mixture-of-experts student, which stores more parameters but activates only a cache-sized subset per token.

## 9. Summary

Speculative decoding turns a large model's spare compute into speed by letting a small model guess ahead. The guesses must agree with the large model often enough to pay for the drafting. On a desktop with an RTX 3070 and a 96 MB-cache CPU, a draft of about 50 million parameters fits in the CPU cache and runs at about 1 ms per token. In llama.cpp's sequential loop, however, the GPU runs such a draft even faster, so the CPU's advantage lies in freeing the GPU rather than in raw speed. Knowledge distillation from the 8B target clearly outperforms ordinary training for this purpose, and starting from a pruned copy of a related small model helps further. After a short pilot the best student agrees with the target 44–59% of the time depending on the domain. Training four times longer raises agreement to 0.56 on English and 0.65 on the financial text, matching the 0.6B model on the latter, and a larger share of structured data in the training mix lifts JSON agreement to 0.89 without hurting prose. Run as a real draft from the CPU, the longer-trained student speeds up the 8B model by 1.18× on English, 1.32× on Portuguese financial text and 1.50× on JSON, beating Qwen3-0.6B on prose and costing only a few percent of speed against running the same draft on the GPU.

## References

- Leviathan, Kalman and Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318, 2023.
- Hinton, Vinyals and Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
- Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation.* ICLR 2024.
- llama.cpp, https://github.com/ggml-org/llama.cpp
