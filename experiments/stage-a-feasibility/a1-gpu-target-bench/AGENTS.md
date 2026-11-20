# a1: GPU target benchmark

**Goal.** Measure the target's per-token decode time t_t, prefill throughput and verification cost t_v(n) for n = 1..8 tokens at batch 1.

**Inputs.** Qwen3-8B Q4_K_M, plus Qwen3-4B Q4_K_M as fallback. llama.cpp at the pinned commit, CUDA build.

**Method.**
- `llama-bench` decode and prefill at context 512 and 2048; then a verification microbenchmark that decodes n tokens in one batch after a fixed prefix.
- Record peak VRAM and free VRAM at start. Close other GPU clients first.
- At least 5 repeats; report warm runs and the first cold run separately.

**Outputs.** t_t, t_v(n) and prefill tokens/s with 95% CIs; VRAM headroom at each context length.

**Note.** The batched verification path may use different CUDA kernels than batch-1 decode. Keep the build flags fixed and record them; c1 measures the numerical effect.
