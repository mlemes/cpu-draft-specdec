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
| Evaluation domains | General English text (Wikipedia), Portuguese financial text (minutes of the Brazilian central bank's monetary policy committee), and structured requests answered in JSON (JavaScript Object Notation, a text format with named fields, such as `{"status": "ok"}`) |

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

4. At the first rejection, a replacement token is sampled from the residual distribution, which favors the tokens the draft gave too little probability, in proportion to the shortfall. The guesses after the rejected one are discarded, because they assumed the rejected token was part of the text.

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

where $\mathrm{TV}$ is the total-variation distance between the two distributions. In words: for each token, take the smaller of the two probabilities and add these up over all tokens. If the sum is 0.8, the target accepts about 80% of the draft's sampled proposals at that position. In greedy decoding, $\alpha$ is the probability that the draft's top token matches the target's top token.

If acceptances are treated as independent with a constant $\alpha$, the number of tokens produced per cycle (accepted drafts plus the one token the target always contributes) has expectation

$$
E(\alpha, \gamma) = \sum_{i=0}^{\gamma} \alpha^i = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}.
$$

With $\gamma = 4$, a draft that agrees half the time ($\alpha = 0.5$) yields $E \approx 1.94$ tokens per cycle; $\alpha = 0.8$ yields $E \approx 3.36$. These are averages. If the second of four guesses is rejected, the cycle emits the first guess plus one replacement token, and the time spent drafting the third and fourth guesses is wasted. Longer blocks therefore pay off only when agreement lasts far enough into the block.

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

- Before any training, a timing test with a block of about 56 MB of weights, the planned size of the draft, took 1.14 ms per token on three CPU cores, with only 3–5% of the touched bytes coming from main memory.
- The same computation, with the cache allocation artificially restricted to 6 MB through the CPU's cache-partitioning feature, took 2.41 ms per token. Cache residency alone is worth a factor of 2.5.

The size limit is strict. An off-the-shelf 0.6-billion-parameter model (Qwen3-0.6B, about 600 MB at 8 bits) does not fit, and on the CPU it *slowed* generation to 0.25–0.47× of plain GPU decoding. The draft must be small, and small models guess poorly unless they are trained for the job.

## 4. Feasibility study

Before training anything, a set of measurements checked whether the idea could work at all.

| Question | Finding |
|---|---|
| Is the target's verify pass cheap? | Yes: 14.2 ms to check five tokens vs 12.6 ms to generate one. |
| Can a draft of the planned size run from the cache? | Yes: about 1.1 ms per token on three cores, with little main-memory traffic. |
| Can the draft's output vocabulary be cut? | Yes. Qwen3 has 151,936 tokens; restricting the draft to the 32,768 the target uses most still covers 96–98% of the target's probability mass, depending on the domain. |
| How well does an off-the-shelf small model agree before any project-specific training? | Qwen3-0.6B, from the same family as the target, reaches $\alpha \approx 0.63$–$0.66$ on prose and 0.99 on JSON. It is too large for the cache, though. |
| Does a cache-sized CPU draft speed anything up in practice? | Yes, with a different model pair. A public 68-million-parameter draft for Llama-2-7B gave 1.40× on English and 1.12× on Portuguese financial text, running on the CPU. |
| Is the CPU better than the GPU for the draft? | No, not in llama.cpp's current loop. The GPU runs the same 68M draft at 0.26 ms per token against 0.63 ms on the CPU, and the GPU version gave 1.65× and 1.32×. |

The last row reshaped the project. llama.cpp's speculative loop is strictly sequential: draft, then verify, then draft again. When the draft runs on the GPU it is simply faster, so a CPU draft cannot win on speed in that loop. A careful breakdown of one cycle confirmed that the gap comes entirely from the draft's own compute time, with no hidden fixed cost.

A CPU draft can still pay off in two ways:

1. **Overlap.** The CPU drafts the next block while the GPU verifies the current one. For the Llama model pair, the sequential CPU-draft cycle took 16.3 ms. If drafting overlapped verification, the projected cycle would take about 13.2 ms, roughly 7% faster than running the draft on the GPU. That is real but modest, and requires changing the engine.
2. **Freeing the GPU.** A draft on the CPU uses no GPU memory and no GPU time. Section 7.6 shows that for a draft this small the memory saved is negligible, so the benefit is GPU time.

The project therefore continued with distillation first and treats the CPU placement mainly as a way to keep the GPU free.

## 5. Distillation

### 5.1 The idea

*Knowledge distillation* trains a small *student* model to imitate a large *teacher*. Ordinary language-model training shows the model a text and asks it to predict each next token; the training signal is a single correct answer per position (a *hard label*). In distillation the teacher first reads the same text, and the student learns to match the teacher's full probability distribution at every position. A distribution such as "the 70%, a 20%, this 5%, …" carries much more information than the single word that happened to appear, including which alternatives are plausible and how uncertain the teacher is.

For speculative decoding this aligns directly with the goal. The acceptance rate is $1 - \mathrm{TV}(p, q)$, so a student whose distribution $q$ is close to the teacher's $p$ is, by construction, a good draft.

### 5.2 The loss

Storing the teacher's full distribution over 151,936 tokens for every training position would take far too much disk. Instead, the teacher's 32 most likely tokens and their probabilities are stored, and everything else is lumped into one *tail* bucket. For each position, the student therefore learns 33 probability values: one per stored token and one for the tail.

Let $K$ be the set of stored tokens, $p_k$ their teacher probabilities, $p_\text{rest} = 1 - \sum_{k \in K} p_k$, and correspondingly $q_k$ and $q_\text{rest}$ for the student. The training loss per position is the forward Kullback–Leibler divergence over this coarsened distribution:

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

Follow-up runs kept the best recipe (distilled, pruned start) and changed one thing at a time. The first two are described here; the table and the text after it add further variants. The first trained four times longer, on 96 million tokens instead of 24 million, about 1 hour on the RTX 3070. Because the scored training text holds only about 25 million tokens, the student saw it about four times. The second run kept the 96 million tokens but raised the share of structured output in the training mix from 8% to 23%, adding about 9,500 function-calling and JSON-mode conversations from the same public collection. None of these overlap the JSON evaluation prompts.

| Student | English | Portuguese financial | JSON |
|---|---|---|---|
| 24 million tokens (section 6) | 0.44 (0.42–0.46) | 0.59 (0.58–0.60) | 0.53 (0.51–0.55) |
| 96 million tokens | 0.56 (0.54–0.58) | 0.65 (0.64–0.67) | 0.75 (0.73–0.77) |
| 96 million tokens, 23% structured output | 0.56 (0.54–0.58) | 0.65 (0.64–0.66) | **0.89 (0.88–0.90)** |
| Same, with 40 million tokens of new prose | 0.57 (0.55–0.59) | 0.65 (0.63–0.66) | 0.89 (0.88–0.90) |
| *Untrained Qwen3-0.6B (8× larger)* | *0.63* | *0.66* | *0.99* |

**Longer training pays off, then saturates.** Four times the training raised agreement by 0.13 on English, 0.06 on the financial text and 0.22 on JSON. The financial text now matches the 0.6B model, and English crosses the rate needed for 1.3×. Most of the gain came in the first 60 million tokens. After that, agreement on prose barely moved.

**More structured data helps structured output at no cost to prose.** The larger share raised JSON agreement from 0.75 to 0.89, still rising at the end of the run, while English and financial text stayed within their confidence intervals. The structured examples replaced repeated passes over prose text that the student had already saturated, so prose lost nothing.

**New prose does not lift the plateau.** A natural explanation for the flat prose curve was repetition: the student saw the same 25 million tokens about three times. A third run replaced most of those repeats with 40 million tokens of new Wikipedia text in both languages, scored by the teacher, so that each prose token was seen about once. Training loss rose, since the student could no longer partly memorize its text, but agreement on held-out prose stayed within the confidence intervals (0.57 against 0.56 on English, 0.65 on the financial text). The plateau near 0.57 on English is therefore a limit of the student itself, most likely its size, and not of the data. Qwen3-0.6B, about eight times larger, reaches 0.63.

**A deeper student helps only a little.** To test the size explanation, a fourth run used 18 layers instead of 12, still pruned from Qwen3-0.6B and still small enough for the CPU cache (about 75 MB of weights read per token at 8-bit precision). English agreement rose from 0.57 to 0.59, within the confidence intervals, and the other domains did not move. Since the deeper draft also takes about 50% longer per token, the cost model predicts that this small gain does not pay for itself, and the speed measurement confirms it: with the draft on the CPU at $\gamma = 2$, the 18-layer draft gives 1.07× on English, 1.17× on the financial text and 1.50× on JSON, against 1.14×, 1.28× and 1.68× for the 12-layer draft measured under the same conditions. It is slower in every domain and at every block length, on the CPU and on the GPU. For a draft, extra capacity is only useful if it does not increase the weights read per token.

## 7. Measured speedups

### 7.1 Running the student inside llama.cpp

llama.cpp expects a draft model to produce a score for every token of the target's vocabulary. The student scores only its 32,768 kept tokens, so a small change to llama.cpp was needed. The model file now carries a map from each vocabulary token to its row in the student's output layer. After computing the reduced scores, the engine expands them to the full vocabulary and gives every token outside the kept set a score of minus infinity, so it can never be drafted. The rest of the engine, including sampling and the speculative loop, is unchanged.

Two checks confirmed the conversion. The engine's probabilities for the converted student match those of the original training code. And in an initial comparison, speculative decoding with the student as the draft produced the same text as the target alone, as the method intends. A broader greedy test (section 7.9) found exact matches on only about half of the prompts. The converted student occupies 139 MB at 8-bit precision. About 82 MB of that is the input embedding table, of which only one row is read per token.

### 7.2 Results

The first speed test used the best student from section 6: pruned start, distilled, trained on 24 million tokens. Sections 7.3 and 7.4 test longer training and more structured data. It was run as the draft for Qwen3-8B in llama.cpp's speculative loop. The prompts and settings were the same as in the earlier measurement with Qwen3-0.6B as the draft: greedy decoding, 64 generated tokens, 100 prompts per domain for $\gamma = 4$ and 30 for $\gamma = 2$ and 6. The baseline is the target alone, at 76 tokens per second.

| Block length $\gamma$ | Draft runs on | English | Portuguese financial | JSON |
|---|---|---|---|---|
| 2 | CPU | 1.04 (1.00–1.07) | **1.27 (1.23–1.31)** | 1.07 (1.03–1.11) |
| 2 | GPU | 1.05 (1.01–1.09) | 1.26 (1.23–1.30) | 1.11 (1.07–1.15) |
| 4 | CPU | 0.92 (0.89–0.94) | 1.18 (1.15–1.21) | 1.02 (0.99–1.05) |
| 4 | GPU | 0.96 (0.93–0.98) | 1.26 (1.23–1.29) | 1.09 (1.06–1.12) |
| 4 | *GPU, Qwen3-0.6B as draft* | *1.03* | *1.09* | *1.90* |

Values are speedups over the target alone, with 95% confidence intervals. Longer blocks ($\gamma = 6$) were slower in every domain.

**Offline agreement tracks the acceptance seen in the engine.** (Wall-clock speedup is measured separately, in the table above.) The engine reports how many drafted tokens it accepted. Converting that count back into a per-position acceptance rate with the formula of section 3.2 gives 0.43 on English, 0.60 on the financial text and 0.51 on JSON. The offline evaluation of section 6 predicted 0.44, 0.59 and 0.53. Students can therefore be compared offline, in minutes, without full speed runs.

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

### 7.4 The structured-output draft

The student trained with 23% structured output (section 6.1) was measured next. Its prose agreement is the same as that of the student in section 7.3, so the interest is JSON.

| Block length $\gamma$ | Draft runs on | English | Portuguese financial | JSON |
|---|---|---|---|---|
| 2 | CPU | 1.14 (1.12–1.16) | 1.27 (1.25–1.30) | 1.68 (1.66–1.70) |
| 4 | CPU | 0.97 (0.95–1.00) | 1.22 (1.18–1.26) | 1.94 (1.90–1.98) |
| 4 | GPU | 1.09 (1.06–1.13) | 1.36 (1.33–1.40) | **2.17 (2.12–2.21)** |
| 6 | GPU | 0.99 (0.94–1.05) | 1.26 (1.19–1.32) | **2.28 (2.17–2.41)** |
| 4 | *GPU, Qwen3-0.6B as draft* | *1.03* | *1.09* | *1.90* |

$\gamma = 6$ used 30 prompts per domain, the others 100. A repeat with 100 prompts at every block length reproduced all rows within 0.08 and gave 2.04× (1.98–2.10) with the draft on the CPU at $\gamma = 6$.

**JSON more than doubles the target's speed.** The acceptance rate implied by the engine's counts is 0.89, exactly the offline prediction. At that rate long blocks pay off, and the best setting is 6 tokens or more. With the draft on the GPU the 8B model runs 2.17× faster at $\gamma = 4$ and 2.28× at $\gamma = 6$, against 1.90× for Qwen3-0.6B. The 50-million-parameter student now beats the off-the-shelf model eight times its size in all three domains.

**On JSON the CPU's slower drafting shows.** With the draft on the CPU the best JSON speedup is 2.04×, about 0.2 below the GPU draft. At an acceptance rate of 0.89 each cycle drafts and keeps many tokens, so drafting time is a larger share of the cycle than on prose. A loop that drafts on the CPU while the GPU verifies would hide most of this cost.

### 7.5 Sampled generation and a loss aimed at acceptance

All results so far use greedy decoding. With sampling, section 3.1 says a drafted token is accepted with probability $\min(1, p/q)$, so the acceptance rate is $1 - \mathrm{TV}(p, q)$ and depends on the draft's whole distribution, not just its top token. The training loss of section 5.2 (forward KL) is not the same objective: it heavily penalizes the student for missing any token the teacher considers possible, which pushes the student to spread probability over the teacher's tail. A loss that directly maximizes the overlap $\sum_x \min(p(x), q(x))$, computed over the teacher's top 32 tokens, targets acceptance itself.

To test this, the best student was trained for 24 million more tokens in two ways, once with the original loss and once with the overlap loss, at a lower learning rate. Greedy agreement was the same for both. The offline estimate of sampled acceptance at temperature 1 rose from 0.49 to 0.54 on English, from 0.63 to 0.65 on the financial text and from 0.85 to 0.87 on JSON with the overlap loss.

The first speed measurement showed no difference, for a practical reason. The draft can propose in two ways. By default, llama.cpp's draft proposes only its most likely token, and the target accepts it when its own sample happens to match. Acceptance then depends only on the draft's top token, and the rest of its distribution is ignored. The rule of section 3.1 is available as an option, in which the draft samples its proposal and the target accepts or rejects it with $\min(1, p/q)$.

| Block length $\gamma$ | Draft proposal | English | Portuguese financial | JSON |
|---|---|---|---|---|
| 4 | Top token (default), original loss | 0.97 | 1.10 | 1.92 |
| 4 | Top token (default), overlap loss | 0.97 | 1.12 | 1.97 |
| 4 | Sampled, accept with $\min(1, p/q)$, original loss | 1.02 | 1.19 | 1.91 |
| 4 | Sampled, accept with $\min(1, p/q)$, overlap loss | **1.06** | **1.24** | 1.96 |

Speedups with the draft on the CPU and sampling at temperature 0.7 (top-k 20, top-p 0.8); confidence intervals about ±0.03.

**The proper acceptance rule matters more than the loss.** Switching the draft to sampled proposals raised sampled speedups by 0.05–0.12 on prose, bringing them close to the greedy speedups. The overlap loss adds a further 0.04–0.05 on prose at $\gamma = 4$. Both gains are smaller than the offline estimate suggests, because at temperature 0.7 with top-k and top-p truncation the target's distribution is sharper and has less of the tail where the two losses differ.

### 7.6 GPU memory

The server's own GPU memory was measured with the benchmark settings (2,048-token context): 5,034 MiB for the target alone, 5,120 MiB with the draft on the GPU, and 5,034 MiB with the draft on the CPU. The GPU draft costs only 86 MiB, because its largest part, the input embedding table, stays in main memory even when the rest of the draft is on the GPU. That is about 600 tokens of the target's attention cache. For a draft this small, the memory saved by moving it to the CPU is negligible; its resource benefit is the GPU time it no longer uses.

### 7.7 Overlapping drafting and verification

In llama.cpp's loop the GPU waits while the draft runs and the CPU waits while the GPU verifies. The cost of each part was measured separately. A target pass on the RTX 3070 takes 14–15 ms whether it checks 1 or 7 tokens, because verification is limited by reading the weights. The draft takes 1.7 ms per token on the CPU. Per step, drafting is therefore 16% of the time at $\gamma = 2$ and about a third at $\gamma = 6$. The remaining step time exceeds draft plus verification by 3–8 ms, spent on sampling, catching the draft up and server bookkeeping.

All table entries below are speedups over the target alone; only the first column is measured, and the other two are projections. For example, on JSON with six drafted tokens the sequential loop reached 2.04×, drafting ahead along one guessed continuation projects to 2.36×, and hiding all drafting time gives an upper bound of 2.94×. Two bounds follow from the measured step times. The simplest overlap keeps drafting during verification on the assumption that every drafted token and the target's extra token will match the draft; the work is useful only when they do, with probability about $\alpha^{\gamma+1}$. An ideal overlap hides the drafting on every step, which would require drafting for several possible outcomes at once; the idle CPU has time for about 8 draft tokens per verification.

| $\gamma$ | Domain | Measured | Draft ahead | Ideal |
|---|---|---|---|---|
| 2 | English | 1.14 | 1.17 | 1.36 |
| 2 | Financial (Portuguese) | 1.28 | 1.33 | 1.53 |
| 4 | JSON | 1.89 | 2.18 | 2.49 |
| 6 | JSON | 2.04 | 2.36 | 2.94 |

**Simple overlap pays only where acceptance is high.** Drafting ahead adds 0.2–0.3 on JSON, where the draft usually matches all tokens, but only 0.01–0.05 on prose. The larger gains on prose need drafting for several outcomes. The fixed per-step overhead is as large as the drafting time at $\gamma = 2$ and is a separate target. These bounds use isolated timings and treat acceptances as independent; real acceptances are correlated, which favours drafting ahead.

### 7.8 A draft-free baseline: n-gram lookup

llama.cpp also offers a proposer that needs no draft model: it looks up the last three tokens in the prompt and the text generated so far, and proposes the tokens that followed the previous occurrence. With up to 4 or 8 proposed tokens, on the same 100 prompts per domain, the speedup over the target alone is about 1.13–1.17× in every domain and in both greedy and sampled generation (1.07–1.11× on sampled general and Portuguese financial text). Acceptance per proposed token is 0.14–0.33, and falls when the proposal is longer.

The comparison is informative in both directions. On prose the n-gram lookup (1.13–1.15× greedy) beats the CPU student at γ = 4 on general text (1.03×) and is close on Portuguese financial text (1.24×), at zero training and memory cost. On JSON the student (1.89× at γ = 4) is far ahead of the lookup (1.14×), so with four-token proposals the distilled draft shows its clearest return on structured output, where the lookup cannot exploit the structure. Section 7.10 shows that other settings shift the prose figures.

### 7.9 Output equality under greedy decoding

Greedy outputs with speculation were compared with target-only greedy outputs on 60 prompts (20 per domain, 128 generated tokens, prompt cache off). With the CPU draft and with the GPU draft, 30 of 60 outputs are identical to the baseline; with the n-gram lookup, 34 of 60. The two draft placements produce the same count, as expected when the output is decided by the verification pass and not by the draft.

The remaining outputs diverge at a median of about 190 characters of roughly 480, and differ from that point on. Speculative decoding as implemented here is therefore not bit-exact against token-by-token decoding. The likely cause is that verification evaluates several positions in one batch, where the matrix kernels accumulate in a different order than in single-token decoding, so near-tied logits can flip the argmax; the flip then changes every later token. This explanation was not isolated in these runs. A direct test would compare the target's logits for the same prefix from the batched and the single-token path, and is left to the planned correctness suite, together with the stochastic distribution tests, the sampler-chain check and the cache rollback check.

### 7.10 Draft thread count

The CPU draft was also run with 3 and 6 threads (4 drafted tokens, greedy, same prompts; 4 threads is the default used above). Speedup over the target alone:

| Draft threads | General | Portuguese financial | JSON |
|---|---|---|---|
| 3 | 0.96× | 1.16× | 1.81× |
| 4 | 1.03× | 1.24× | 1.89× |
| 6 | 1.09× | 1.31× | 2.07× |

Acceptance (the share of proposed tokens the target accepted, as reported by the engine) is the same in every row (0.26, 0.38, 0.76), so the whole difference is draft latency: more threads shorten each drafting burst, and the GPU verification does not need the cores. The 6-thread setting raises the JSON speedup to 2.07× and lifts general text above break-even. With 6 threads the draft length can be revisited:

| Drafted tokens | General | Portuguese financial | JSON |
|---|---|---|---|
| 2 | 1.22× | 1.34× | 1.75× |
| 4 | 1.09× | 1.31× | 2.07× |
| 6 | 0.97× | 1.19× | 2.26× |

The best length depends on the domain: 2 tokens on prose, 6 or more on JSON, where acceptance stays at 0.70. A fixed length leaves speedup on the table, which motivates an adaptive draft length. The machine has six physical cores, so 6 threads leaves none for the server's own thread; whether that costs anything under longer prompts was not measured.

## 8. What remains

1. **Better use of the student's size.** Neither new text nor 50% more depth moved English agreement much. Remaining options are a sparser design that reads fewer weights per token (below) and training objectives aimed directly at acceptance rather than at matching the full distribution.
2. **Measure the value of freed GPU time.** The memory saving is negligible (section 7.6); the CPU draft's remaining resource benefit is GPU time, which matters when the card is shared, for example with a second model or several requests at once.
3. **Overlap drafting and verification.** Drafting ahead during verification would add 0.2–0.3 on JSON (section 7.7); on prose it needs drafting for several outcomes, and the fixed per-step overhead of the loop is a comparable target.
4. **Adaptive draft length.** The best number of drafted tokens differs by domain (section 7.10): 2 on prose, 6 or more on JSON. Choosing it per request or per step from recent acceptance is untested.
5. **Exactness and sampling tests.** Greedy outputs match target-only decoding on only half of the prompts (section 7.9). The stochastic distribution tests, the sampler-chain check and the cache rollback check remain.
6. **Wider sweeps.** Longer contexts, 300 million training tokens and 8-bit versus 4-bit draft weights are unmeasured.
7. **Conditional computation.** The original plan also compares the dense student with a mixture-of-experts student, which stores more parameters but activates only a cache-sized subset per token.

## 9. Summary

In short, a distilled draft of about 50 million parameters running on the CPU speeds up the 8B model on the GPU, most on structured JSON (about 2×). Output from speculative decoding matched target-only greedy output on only about half of the test prompts (section 7.9), so exactness is not yet established.

Speculative decoding turns a large model's spare compute into speed by letting a small model guess ahead. The guesses must agree with the large model often enough to pay for the drafting. On a desktop with an RTX 3070 and a 96 MB-cache CPU, a draft of about 50 million parameters fits in the CPU cache and runs at about 1 ms per token. In llama.cpp's sequential loop, however, the GPU runs such a draft even faster, so the CPU's advantage lies in freeing GPU time rather than in raw speed; the GPU memory it saves is only about 86 MiB.

Knowledge distillation from the 8B target clearly outperforms ordinary training for this purpose, and starting from a pruned copy of a related small model helps further. After a short pilot the best student agrees with the target 44–59% of the time depending on the domain. Training four times longer raises agreement to 0.56 on English and 0.65 on the financial text, matching the 0.6B model on the latter, and a larger share of structured data in the training mix lifts JSON agreement to 0.89 without hurting prose.

Run as a real draft from the CPU, the longer-trained student speeds up the 8B model by 1.18× on English, 1.32× on Portuguese financial text and 1.50× on JSON, beating Qwen3-0.6B on prose and costing only a few percent of speed against running the same draft on the GPU. Trained with more structured data, the draft makes the 8B model more than twice as fast on JSON.

Overlapping CPU drafting with GPU verification could raise JSON further, to about 2.2–2.4×, but adds little on prose unless the draft prepares for several outcomes at once. A draft-free n-gram lookup gives about 1.13–1.17× in every domain (section 7.8), so the student pays off mainly on JSON (1.89× against 1.14×). Giving the CPU draft six threads instead of four raises the speedups to 1.09× on English, 1.31× on Portuguese financial text and 2.07× on JSON, and the best draft length varies by domain, reaching 2.26× on JSON at six tokens. Greedy outputs with speculation match target-only outputs on about half of the test prompts, diverging mid-text, which points to numerical differences in the batched verification pass.

## References

- Leviathan, Kalman and Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318, 2023.
- Hinton, Vinyals and Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
- Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation.* ICLR 2024.
- llama.cpp, https://github.com/ggml-org/llama.cpp
