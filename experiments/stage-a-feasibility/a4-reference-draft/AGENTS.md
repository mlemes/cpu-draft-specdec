# a4: reference draft (Qwen3-0.6B)

**Goal.** Measure acceptance and timing for the official same-tokenizer draft, as a negative control and as an upper reference for what a small student might reach.

**Inputs.** Qwen3-0.6B (Q8_0 and Q4_0) and the a1 target. Prompt sets: general, Portuguese financial, structured JSON, 100 prompts each, fixed and hashed.

**Method.**
- Draft on CPU and, separately, on GPU, with the target on GPU; γ = 2, 4, 6.
- Greedy and T = 0.7 with the fixed sampler chain.
- Record per-token acceptance α, mean accepted length, per-round time split, and touched bytes per token (expected about 630 MB at Q8_0, DRAM-bound).

**Outputs.** α per domain and decoding mode with CI; measured speedup; breakdown of round time.

**Use.** If Qwen3-0.6B acceptance is already below the α needed at the small student's expected t_d, the student must be unusually well aligned. Flag this in the Stage A summary.
