# c2: baselines

**Goal.** Measure every comparison point with the same target file, quantization, sampler chain and prompt sets as c3.

**Baselines.**
1. Optimized GPU target alone (the denominator of every speedup).
2. The same student draft on the GPU (it fits easily in VRAM).
3. n-gram or prompt-lookup proposer (llama.cpp lookup decoding, or a suffix-based proposer).
4. Standalone CPU student, for its quality–latency tradeoff (from b6).

**Outputs.** Tokens/s, time to first token, p50 and p95 latency, and committed tokens per round, each with CI.
