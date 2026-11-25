# CPU draft, GPU target: cache-resident speculative decoding

Research proposal. Written 2026-11-20. Status: design only, no experiment run yet.

## Evidence labels

Every factual claim in this project carries one label.

- **[V]** Verified on this machine or by opening the primary source.
- **[R]** Cited from a primary source by a delegated researcher, not yet opened by the author. Treat it as a lead to confirm before it supports a result.
- **[E]** Estimate. The assumptions are stated next to it.
- **[U]** Unknown. A stage below must resolve it.

## 1. Objective

Distill an autoregressive language model that runs on an RTX 3070 into a small CPU draft model for speculative decoding. The target model stays on the GPU and verifies the CPU's proposals.

The student may store more parameters than the CPU cache holds; inactive parameters can stay in RAM. The project asks whether a small, reusable set of active parameters can stay resident in the 96 MB L3 cache while keeping enough draft–target agreement to accelerate generation. The student is also evaluated on its own, with standalone quality reported separately from speculative decoding.

Research questions, treated as hypotheses and not promised outcomes:

1. Can a distilled CPU draft accelerate GPU inference after drafting, verification, communication and synchronization are all counted?
2. At a fixed active-weight budget, can conditional computation beat a small dense student?
3. Does temporal reuse of active parameters produce measurable cache benefits?
4. Under changing workloads, when do inference savings justify refreshing the student?

Useful negative results are in scope. The project should stop or narrow when the gates in section 8 fail.

## 2. Hypotheses and relation to prior work

**H1 (primary).** At batch size 1, with the target checkpoint, quantization and decoding settings fixed (Qwen3-8B, Q4_K_M), a distilled CPU draft can raise end-to-end decode tokens per second by at least 1.3× over optimized GPU-only decoding. The draft must touch at most 60 MB of weight bytes per generated token, counting a pruned output head. Its per-token latency should rise by at least 1.5× when the L3 allocation is reduced below its measured working set with Cache Allocation Technology (CAT), while its compute stays the same.

H1 is falsified if the lower bound of the 95% confidence interval of the speedup is at or below 1.0, or if latency and DRAM bytes per token show no knee as the CAT allocation shrinks.

**H2.** At matched active bytes per token, a mixture-of-experts (MoE) student that stores about 3× more parameters reaches a higher acceptance rate α than the dense student. H2 is falsified if Δα ≤ 0.02 inside the confidence interval.

**H3.** Routing reuse across a speculative block keeps the MoE student's DRAM traffic per token at or below 25% of its stored bytes. A sticky-routing regularizer should move this metric, and a stack-distance model built from routing traces should predict the traffic that the hardware counters measure.

**H4.** An update to the student pays off only when the remaining workload is longer than N* = C_update / Δt_per_token (defined in Stage D).

**H5 (added 2026-11-24, exploratory).** At the touched bytes per token of D-S, a student whose FFN is a Kolmogorov-Arnold block (B-spline KAN or group-rational KAN) reaches a higher acceptance rate α than D-S and than a plain MLP of equal bytes. H5 is falsified if no KAN variant beats both outside the confidence interval. Prior evidence points against it [R: Yu et al. 2024], so a negative result is the expected outcome [E].

### Prior work

| Work | What it contributes here |
|---|---|
| Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding" | Lossless rejection sampling; expected committed tokens per round E = (1 − α^(γ+1)) / (1 − α). |
| DistillSpec (ICLR 2024) | Knowledge distillation (KD) for draft alignment; total-variation (TV) objective matches acceptance; on-policy, student-generated data. |
| Online Speculative Decoding (ICML 2024) | Updates the draft from data produced during verification. |
| FastDraft (arXiv 2411.11055) [R] | Draft training with both draft and target on an Intel CPU. |
| Dovetail (arXiv 2412.18934, EMNLP 2025) [R] | Heterogeneous CPU/GPU speculative decoding with the opposite placement: draft on the GPU, target verifying on the CPU. |
| FR-Spec (arXiv 2502.14856), VocabTrim (arXiv 2506.22694) [R] | Pruning the draft's output vocabulary to cut LM-head cost. |
| MoE-Spec (arXiv 2602.16052) [R] | Expert budgeting on an MoE verifier, not an MoE draft. |
| EAGLE-3, HASS, GRIFFIN [R] | Draft heads and alignment objectives conditioned on target features. |
| PowerInfer, Deja Vu, TEAL, Fiddler, HOBBIT [R] | Locality of sparse activations and experts, studied for large models. |
| SuffixDecoding, REST, prompt lookup [R] | Model-free proposers, used here as baselines. |

**Gap.** No located work runs a CPU-resident neural draft feeding a GPU verifier at batch 1, and none measures L3 residency of a draft's active weights. A later search should re-check this before writing.

## 3. Environment (inspected 2026-11-20)

| Item | Finding | Label |
|---|---|---|
| OS | Ubuntu 26.04.1, Linux 7.0.0-43-generic | [V] |
| CPU | Ryzen 5 5500X3D, family 25 model 33, 6 cores / 12 threads, boost to 4.05 GHz, `amd-pstate`, governor `performance`, SMT on | [V] |
| Caches | L1d 32 KiB and L2 512 KiB per core; one L3 of 96 MiB shared by all 6 cores, 16-way, 64 B lines (6 MiB per way) | [V] |
| Instruction set | AVX2, FMA, F16C, BMI2, SHA. No AVX-512, no AVX-VNNI | [V] |
| Cache control | Flags `cat_l3`, `cdp_l3`, `mba`, `cqm_occup_llc`, `cqm_mbm_total`, `cqm_mbm_local`, `ibs`, `perfctr_llc`. Kernel has `CONFIG_RESCTRL_FS=y`; resctrl not mounted | [V] |
| RAM | 30 GiB total, 18 GiB available, swap 7.0 of 8 GiB used (background load). DIMM speed unknown without sudo | [V]/[U] |
| Profiling | `perf_event_paranoid=4`. `amd_uncore` module present but not loaded, so `amd_l3` and `amd_df` PMUs are absent. `perf`, `nsys`, `ncu` installed. No AMD uProf, no likwid | [V] |
| GPU | RTX 3070, 8 GiB, driver 595.99.02, CUDA 12.4 (`nvcc`). About 2 GiB in use by other processes, so about 6 GiB free | [V] |
| PCIe | Board ASRock B450M Steel Legend; link max Gen3 x16, although the 3070 supports Gen4 | [V] |
| Software | No system torch, no llama.cpp build. `uv` 0.12.13 available | [V] |
| Disk | `/` has 38 GB free; `/mnt/hdd1` has 615 GB free (HDD) | [V] |
| Local models | `/mnt/hdd1/local-coder-llm/models/` holds Qwen3.5-9B Q4_K_M and others; no Qwen3-8B yet. Qwen3.5-9B GGUF: vocab 248,320, `output.weight` 796 MiB (Q6_K), `token_embd` 546 MiB (Q4_K) | [V] |

Prerequisites that need root, not yet done:

1. `sudo sysctl kernel.perf_event_paranoid=-1` (or grant CAP_PERFMON to the benchmark binary).
2. `sudo modprobe amd_uncore`.
3. `sudo mount -t resctrl resctrl /sys/fs/resctrl`.
4. `echo off | sudo tee /sys/devices/system/cpu/smt/control` during benchmark runs.
5. Run headless, or close GPU clients, to free VRAM and remove noisy neighbours.

## 4. Defining the memory claim

The claim "the active set fits in L3" is defined in bytes of the real runtime representation, after any repacking, and never by parameter count or checkpoint size. Each run reports the following quantities separately.

1. **Stored bytes.** All weight tensors plus quantization metadata (block scales) as loaded, including any heap buffer created by repacking.
2. **Touched bytes per token.** Unique weight bytes read while producing one token: every attention projection, the FFN (dense, or shared plus selected experts), all norms, the output projection over the draft vocabulary, and one input-embedding row. Computed from the tensor layout and routing decisions, then checked against counters.
3. **Block union.** Unique weight bytes touched across the γ draft tokens of one speculative round plus the correction token.
4. **Reuse and replacement.** Per-layer Jaccard similarity of expert sets between consecutive tokens, and the reuse-distance histogram in bytes.
5. **Non-weight state.** KV cache, activations, logits, routing state, scratch buffers and runtime overhead, reported apart from weights.

"Active weights fit in L3" and "the complete inference working set fits in L3" are separate claims. KV state may live in RAM, but its traffic and its interference with weights are measured.

### The output projection dominates

| Vocabulary | Hidden size | Q8_0 | Q4_0 |
|---|---|---|---|
| 151,936 (Qwen3, full) | 512 | 83 MB | 44 MB |
| 151,936 | 1024 | 165 MB | 88 MB |
| 32,768 (pruned) | 512 | 18 MB | 9 MB |
| 32,768 | 1024 | 36 MB | 19 MB |

Q8_0 costs 34 bytes per 32 weights; Q4_0 costs 18 bytes per 32 weights. A full-vocabulary head alone fills or exceeds L3, so the draft's output head is restricted to a frequency-ranked subset V′ of the target vocabulary (as in FR-Spec and VocabTrim). Input embeddings keep the full vocabulary: they are stored in RAM but cost one row per token. Token IDs stay identical to the target's.

Pruning the head caps acceptance: α ≤ p(V′), the target's expected probability mass inside V′. This bound is measured before any training (experiment a5).

## 5. Models and runtimes

### Target

- **Primary: Qwen3-8B, Q4_K_M** (about 5 GB [E]). Pure attention, so KV rollback is simple truncation. Same tokenizer as the official Qwen3-0.6B. Qwen3 claims coverage of more than 100 languages [R]. KV cache: 36 layers × 8 KV heads × 128 dims × 2 (K, V) × 2 bytes = 147 KB per token, about 0.56 GiB at 4k context.
- **Fallback: Qwen3-4B, Q4_K_M** (about 2.5 GB [E]).
- **Rejected for now: Qwen3.5-9B.** Its head is 1.63× larger (vocab 248,320 [V]). If its architecture includes recurrent layers, rollback needs state snapshots [U].

### Drafts

| Name | Architecture | Bytes |
|---|---|---|
| Reference: Qwen3-0.6B (official) | Hidden 1024, 28 layers, vocab 151,936, tied embeddings [R] | Body 468 MB (Q8_0) or 248 MB (Q4_0), plus 165 MB head. DRAM-bound by construction; serves as a negative control. |
| D-S, dense student | Qwen3 layout: hidden 512, 12 layers, FFN 1536 (SwiGLU), 8 query heads / 2 KV heads, head dim 64. Full input embedding; untied output head over V′ = 32,768 | Touched per token: 56 MB (Q8_0), 30 MB (Q4_0). Stored input embedding 83 MB (Q8_0), of which about 0.5 KB is touched per token. |
| M, conditional student | Same attention as D-S. FFN = one shared expert of width 768 plus 16 routed experts of width 384, top-2 routing | Active per token 56 MB (Q8_0), matched to D-S. Stored 162 MB. Worst-case block union at γ = 4 (up to 8 distinct routed experts per layer): 101 MB. One routed expert ≈ 627 KB (Q8_0). |
| D-L, dense control | Dense model sized to about 150–160 MB stored | Separates "more total parameters" from "conditional computation". |

Student KV cache: 12 layers × 2 KV heads × 64 dims × 2 × 2 bytes = 6 KB per token, so 25 MB at 4k context. This competes with weights for L3, which is why context length is a controlled variable.

### Initialization and training

- **D-S** comes from structured depth and width pruning of Qwen3-0.6B, then KD. A D-S trained from scratch with the same token budget is the control for the initialization.
- **M** is upcycled from D-S. The dense FFN's 1536 neurons are split into four groups of 384; two groups form the shared expert, the other two are each replicated with small noise into eight experts (sixteen in total), and the router starts near uniform. M is therefore approximately equal to D-S at initialization, and both are continued for the same number of tokens.
- **Why M is trainable here.** It has 152M parameters; bf16 weights, fp32 master weights and Adam state total about 2.4 GB, which fits on the GPU with activation checkpointing. The cost is slower expert dispatch in PyTorch, estimated at 2–3× the dense student's time [E].

### Runtime

- **llama.cpp, one process**, with the target on CUDA and the draft on CPU (`-md`, `-ngld 0`, `--device-draft`) [R].
- Stochastic verification is implemented in `common_sampler_sample_and_accept_n_rejection` (`common/sampling.cpp`) [R]; Stage C audits it before relying on it.
- The CPU MoE kernel (`ggml_compute_forward_mul_mat_id`) skips experts with no routed tokens [R].
- AVX2 repacked kernels exist for Q4_0 (8×8 interleave), Q4_K and IQ4_NL [R; the Q4_0 AVX2 kernel was also found in a commit search]. Repacking allocates a heap buffer in place of the memory-mapped file [R], so repacked bytes are what Stage A measures. Q8_0 reportedly has no AVX2 repack [R, unconfirmed by search].
- **Required patch:** an output projection restricted to V′ plus an index map back to target token IDs, for the draft only.
- Shared-expert support differs between llama.cpp's qwen3moe and qwen2moe architectures [U]; resolve before fixing M's tensor layout.

### Distillation stack

- PyTorch with a custom trainer (chunked loss over the head so full-vocabulary logits never materialize), environment managed with `uv`.
- Arcee DistillKit supports offline top-k teacher logits [R]; TRL's distillation trainers and torchtune's KD recipe expect an online teacher [R].
- **Teacher logits come from the deployed Q4_K_M target**, because acceptance is defined against the deployed distribution. A bf16 8B teacher does not fit in 8 GB.
- The RTX 3070 (Ampere, sm_86) supports bf16 [R].

## 6. Budgets

### Latency and break-even

A round drafts γ tokens and verifies them in one target pass:

- T_round = γ·t_d + t_v(γ) + t_comm + t_sync
- Speedup S = E · t_t / T_round, with E = (1 − α^(γ+1)) / (1 − α)

Assumptions [E]: target decode t_t ≈ 15 ms per token (memory floor 5 GB ÷ 448 GB/s ≈ 11 ms); t_v(γ) ≈ t_t · (1 + 0.05γ); t_comm + t_sync ≈ 0.3 ms.

Acceptance α needed for S = 1.0 / S = 1.3:

| t_d per draft token | γ = 2 | γ = 4 | γ = 6 |
|---|---|---|---|
| 1 ms | 0.21 / 0.44 | 0.33 / 0.50 | 0.42 / 0.56 |
| 2.5 ms | 0.34 / 0.57 | 0.49 / 0.64 | 0.58 / 0.70 |
| 5 ms | 0.52 / 0.76 | 0.66 / 0.79 | 0.73 / 0.83 |
| 16 ms (≈ Qwen3-0.6B Q8_0 streamed from DRAM) | not reachable | not reachable | not reachable |

Stage A replaces every assumption in this table with measurements.

Reference figures for related parts: 5800X3D L3 about 11–14 ns and 1.5–2 TB/s aggregate; DDR4-3200 about 65–75 ns and 38–44 GB/s sustained [R]. The 5500X3D has two fewer cores, so its aggregate L3 bandwidth is unknown [U]. PCIe 3.0 small pinned copies take about 5–12 µs [R, not measured].

A structural caveat: a draft this small also fits trivially on the GPU (56 MB), where it is limited by kernel-launch overhead. The CPU placement wins only if its per-token cost is lower, if VRAM is exhausted by the target and its KV cache, or if drafting overlaps with verification. Stage C measures the GPU-resident draft as a baseline for this reason.

### Storage and training

| Item | Estimate [E] |
|---|---|
| Teacher top-32 record | 32 × (4-byte token ID + 2-byte fp16 logit) + 2-byte tail mass ≈ 194 B per token |
| 100M teacher tokens | ≈ 19 GB on `/mnt/hdd1` |
| Teacher scoring (prefill) | ≈ 2k tokens/s, so ≈ 14 h per 100M tokens |
| D-S training | ≈ 3 × 10⁸ FLOP per token at ≈ 6 effective TFLOPS ⇒ ≈ 15 h per 1B tokens |
| M training | 2–3× D-S per token |
| Peak VRAM during training | Teacher and student never share the GPU; teacher data is generated first, then training runs alone |

Online teacher plus student training does not fit in the roughly 6 GB of free VRAM, so the pipeline is sequential.

### Approximations in the distillation targets

- **Truncation.** With top-k teacher logits, forward KL uses a single tail bin for the remaining mass. For the TV objective, Σ min(p, q) is exact on the top-k tokens; for tokens outside the top-k, the contribution is bounded below by 0 and above by min(p_k, q), where p_k is the k-th largest teacher probability. Both bounds are reported.
- **Vocabulary pruning.** Teacher mass outside V′ is unreachable for the draft, so α ≤ p(V′).

## 7. Decoding correctness

The target checkpoint, quantization and decoding settings are fixed across every comparison.

### Greedy

The longest draft prefix that matches the target's argmax is accepted. The target's token at the first mismatch is appended, or the bonus token if every draft token matched. Only token IDs cross PCIe.

Batched verification kernels (n > 1 tokens) can choose a different argmax than batch-1 decoding because of floating-point differences. The preserved distribution is therefore defined as the target's distribution along the verification path. The argmax flip rate and the KL divergence between the verify-path and decode-path distributions are measured and reported.

### Stochastic

1. The identical sampler chain (temperature, top-k, top-p, min-p, penalties computed from the current prefix) produces the target distribution p̃ and the draft distribution q̃.
2. Each draft token x is accepted with probability min(1, p̃(x) / q̃(x)).
3. At the first rejection, a token is sampled from norm(max(0, p̃ − q̃)).
4. If all γ tokens are accepted, a bonus token is sampled from p̃ at position γ + 1.

The residual step needs full distributions; selected-token probabilities are not enough. Two placements are measured:

- (a) Target logits copied to the CPU: 4 B × 151,936 × (γ + 1) ≈ 3 MB per round at γ = 4, about 0.25 ms on PCIe 3.0 [E].
- (b) q̃ over V′ sent to the GPU (fp16, about 256 KB per round at γ = 4), verification kernel on the GPU, only the accepted count and sampled token returned.

### KV rollback

Both models drop cache positions past the accepted length (`seq_rm`). The draft then decodes the correction or bonus token before the next round.

### Tests

- Exact enumeration of the accept/reject procedure on a toy vocabulary with known p and q.
- Chi-square and TV tests comparing the distribution of the first k generated tokens with and without speculative decoding, on fixed prompts and seeds.

## 8. Staged experiment plan

| Stage | Configurations | Controls | Runtime [E] | Stop if |
|---|---|---|---|---|
| A. Feasibility | Target decode and prefill speed, and verification cost for n = 1..8 tokens. CPU latency of random-weight GGUFs from 16 to 512 MB, Q4_0 and Q8_0, at 1, 3 and 6 threads with SMT off, under CAT allocations of 1, 2, 4, 8 and 16 ways. Pinned H2D/D2H copies from 1 KB to 3 MB. Qwen3-0.6B acceptance on CPU and GPU, greedy and T = 0.7, three domains × 100 prompts. Teacher coverage p(V′) for 16k, 32k and 64k tokens. | Counter validation comes first: a streaming buffer of size S must show MBM bytes ≈ S when S exceeds the CAT allocation and ≈ 0 when it fits. Other processes are confined to disjoint ways. | 2–3 days | t_d for a ≤ 60 MB model at γ = 4 is at least 5 ms; or latency and MBM bytes per token show no knee as ways shrink (then drop the cache claim and reframe as "small CPU draft"); or p(V′ = 32k) < 0.9. |
| B. Distillation and cache | D-S with hard labels; D-S with forward-KL KD on top-32 targets; D-S with TV KD on student-generated data; M with and without a sticky-routing penalty; D-L. 300M tokens each. | Equal token budgets. M upcycled from D-S, compared with D-S continued for the same tokens. CAT varies cache capacity with compute fixed. Context fixed at 256 and 2048. Q4 and Q8 reported separately, since precision changes both bytes and dequantization cost. Thread count fixed. | ≈ 5 days | D-S α is below the break-even α for its measured t_d minus 0.05: stop the neural CPU draft, report the negative result, narrow to a GPU draft or a hybrid with prompt lookup. H2 is rejected if M's Δα ≤ 0.02, or if its DRAM bytes per token erase its latency advantage. |
| C. End-to-end | Batch 1. Context 512 and 2048. γ = 2, 4, 6. Threads 3 and 6. A chosen subset of combinations, not a full grid. Optional: the CPU drafts the next block while the GPU verifies. | GPU target alone; CPU draft with GPU target; the same draft on the GPU; n-gram or prompt-lookup proposer; standalone CPU student for its quality–latency tradeoff. Same target checkpoint, quantization and sampler chain everywhere. | ≈ 2 days | The speedup's 95% CI lower bound is ≤ 1.0 for every draft: report the negative result. |
| D. Adaptation economics | Held-out streams of general text, Portuguese financial text (public central bank and securities-regulator documents) and structured JSON outputs, in segments of 20k–200k tokens with stable periods, abrupt switches and recurrences. Policies: frozen, periodic, change-triggered (CUSUM on α), and an amortization rule (update when expected remaining tokens × predicted Δ exceeds C_u). | C_u counts teacher scoring, training, quantization and repacking, reload, cache warm-up and lost GPU serving throughput. Teacher data captured during verification (as in Online Speculative Decoding) has near-zero generation cost. Prior student versions are kept for recurring workloads. | 3–4 days | N* exceeds typical segment length: the frozen student wins. Cumulative net seconds saved are reported either way. |

### Design change 2026-11-24: KAN feed-forward arms (experiment b10)

Adds H5 as an exploratory Stage B arm. The operator asked whether Kolmogorov-Arnold networks make a better student at the same budget.

- **Matched quantity.** Touched weight bytes per token at Q8_0, as everywhere else (section 4). The arms do not share widths: a KAN edge stores G + k + 1 numbers (9 at G = 5, k = 3), so a KAN at D-S widths would read about 9× the FFN bytes [E]. Matched widths: B-spline KAN 512-256-512; two-matrix MLP, rational MLP and GR-KAN 512-2304-512. All are within 0.01% of D-S's 56.3 MB [V, arithmetic in `experiments/stage-b-distillation/b10-kan-ffn/configs/budget.py`].
- **Scope.** Only the FFN changes. Every arm copies attention, norms, embeddings and the V′ head from the trained b5 D-S checkpoint and trains 96M more tokens; controls are D-S continued unchanged and D-S with its FFN re-initialized. The arms form a chain where each changes one property: gate removed, then learnable activation, then rational on both layers' inputs, then per-edge splines.
- **"KAN approximating the student's layers"** becomes an initialization variant: each KAN FFN is first fitted to the trained D-S FFN on D-S hidden states, then trained with the same KD run. Fitting each layer on its own without end-to-end training would compound errors across layers, so it is not an arm by itself.
- **Budget.** 96M tokens per arm (the b5/b6 budget), offline greedy agreement first. The llama.cpp runtime path (basis-expansion or rational ops plus a standard quantized matmul) is built only if a KAN arm passes. Draft latency on AVX2 is [U] until then.
- **Order.** Runs after the D-S gate, on a dedicated GPU, and does not delay Stage C.

### Adaptation cost model

For an update u:

- C_u = t_teacher + t_train + t_quantize + t_reload + t_warmup + t_serving_lost, where GPU time taken from serving is converted to lost tokens at the target's throughput.
- Δ = t_per_token(old draft) − t_per_token(new draft) on the new workload.
- Break-even N* = C_u / Δ tokens.

## 9. Measurement protocol

### Reported metrics

- End-to-end tokens/s, time to first token, p50 and p95 per-token latency.
- Mean accepted draft length and committed tokens per verification round.
- Time split into drafting, verification, transfer and synchronization.
- Touched, block-union and stored bytes (section 4), total model memory, peak VRAM.
- Standalone student quality: perplexity on held-out text, top-1 agreement and TV distance with the teacher, and a small task set.
- Adaptation cost and cumulative net savings.

### Hardware counters and their limits

| Source | Use | Limitation |
|---|---|---|
| Core events `ls_dmnd_fills_from_sys.{int_cache, mem_io_local}` plus hardware-prefetch fill events | Per-thread demand fills from L3 versus DRAM | Prefetch fills must be counted too, or DRAM traffic is undercounted [R] |
| resctrl MBM `mbm_total_bytes` | DRAM bytes per token for the benchmark's monitoring group; main evidence for residency | Group-level, not per tensor [R] |
| resctrl `llc_occupancy` | L3 capacity held by the group | Counts all of the group's lines, not only weights [R] |
| `amd_l3` uncore | L3 lookups and misses | CCX-wide, not per process; cannot sample [R] |
| `amd_df` | DRAM channel events | Byte events may read 0 on desktop Zen 3; no UMC PMU before Zen 4 [R] |
| IBS op sampling (`perf mem`) | Attributing misses to tensors | Sampling bias; no hardware L3-miss-only filter on Zen 3 [R] |

Counters are trusted only after the Stage A positive control passes. Measured DRAM traffic is compared with a stack-distance model built from routing traces.

### Cache-capacity controls

- **CAT ways sweep** is the causal lever: it changes effective cache capacity while compute, context and precision stay fixed.
- **Model-size sweep** with random weights changes compute too, so it is reported per byte and per FLOP.
- **Context length** grows the KV cache, which competes for L3; fixed at 256 and 2048.
- **Precision** changes bytes and dequantization work together; Q4 and Q8 are reported separately.
- **Threads** change bandwidth demand; fixed per comparison.

### Experimental discipline

- Prefill and decode reported separately.
- Cold runs start after streaming a 192 MB buffer through the cache; warm runs follow two warm-up prompts.
- At least five repeats, fixed workloads, fixed seeds, paired per-prompt bootstrap 95% confidence intervals for ratios.
- Every run records a manifest ("Run manifest" in the root `AGENTS.md`).

## 10. Smallest useful pilot

Stage A only, about three days, no training:

1. Random-weight size sweep crossed with the CAT ways sweep, giving the t_d(bytes) curve and the cache knee.
2. Target verification cost curve and PCIe timings.
3. Qwen3-0.6B acceptance with the draft on CPU and on GPU.
4. Teacher coverage p(V′).

**Decision it resolves:** whether any feasible (γ, t_d, α) region exists on this machine, and whether L3 residency is measurable at all. Expected outcome [E]: Qwen3-0.6B on the CPU is a negative control, DRAM-bound and below 1×. The project then rests on whether a student of about 50M parameters can reach α of roughly 0.5–0.64. That is the largest unknown, and the Stage B gate tests it directly.

If there is no CAT knee, drop the cache claim. If there is no confidence-interval-backed speedup, publish the measured negative result as a bound on the design space.

## 11. Main risks

- **Acceptance of a tiny student.** A model of about 50M parameters may agree with an 8B target too rarely to clear the break-even α.
- **GPU draft dominance.** A 56 MB draft on the GPU may beat the CPU draft on the same machine, removing the case for CPU placement except under VRAM pressure or overlap.
- **Counter ambiguity.** Zen 3 desktop counters may be too coarse; MBM and the CAT sweep carry the cache argument.
- **Background load.** The machine runs other workloads (swap is nearly full); benchmarks need a quiet session.
- **Runtime support.** The V′ head patch and the MoE shared-expert layout require changes to llama.cpp.

## 12. Sources

Starting references:

- Leviathan et al. 2023: https://proceedings.mlr.press/v202/leviathan23a.html
- DistillSpec: https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html
- Online Speculative Decoding: https://proceedings.mlr.press/v235/liu24y.html

Newer work [R unless noted]:

- Dovetail: https://arxiv.org/abs/2412.18934 (EMNLP 2025: https://aclanthology.org/2025.emnlp-main.879; placement confirmed by search)
- MoE-Spec: https://arxiv.org/abs/2602.16052 (abstract confirmed by search)
- FastDraft: https://arxiv.org/abs/2411.11055
- EAGLE-3: https://arxiv.org/abs/2503.01840
- HASS: https://arxiv.org/abs/2408.15766
- GRIFFIN: https://arxiv.org/abs/2502.11018
- FR-Spec: https://arxiv.org/abs/2502.14856
- VocabTrim: https://arxiv.org/abs/2506.22694
- PowerInfer: https://arxiv.org/abs/2312.12456
- Deja Vu: https://arxiv.org/abs/2310.17157
- TEAL: https://arxiv.org/abs/2408.14690
- Fiddler: https://arxiv.org/abs/2402.07033
- HOBBIT: https://arxiv.org/abs/2411.01433
- SuffixDecoding: https://arxiv.org/abs/2411.04975
- REST: https://arxiv.org/abs/2311.08252
- SpecExec: https://arxiv.org/abs/2406.02532

Kolmogorov-Arnold networks (b10) [R; titles and authors confirmed by search 2026-11-24]:

- KAN (Liu et al. 2024): https://arxiv.org/abs/2404.19756
- Kolmogorov-Arnold Transformer, GR-KAN (Yang and Wang, ICLR 2025): https://arxiv.org/abs/2409.10594
- KAN or MLP: A Fairer Comparison (Yu, Yu and Wang 2024): https://arxiv.org/abs/2407.16674
- FastKAN, KANs as RBF networks (Li 2024): https://arxiv.org/abs/2405.06721
- FlashKAT, KAT training bottlenecks (Raffel and Chen): https://arxiv.org/abs/2505.13813

Models, runtimes and tools [R]:

- Qwen3-0.6B config: https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- Qwen3-30B-A3B config: https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json
- llama.cpp sampling: https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp
- llama.cpp CPU repack: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.cpp
- llama.cpp AVX2 Q4_0_8_8 commit: https://app.semanticdiff.com/gh/ngxson/llama.cpp/commit/581c305186a0ff93f360346c57e21fe16e967bb7
- ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp
- DistillKit: https://github.com/arcee-ai/DistillKit
- TRL distillation trainer: https://github.com/huggingface/trl/blob/main/trl/trainer/distillation_trainer.py
- Linux resctrl documentation: https://docs.kernel.org/arch/x86/resctrl.html
