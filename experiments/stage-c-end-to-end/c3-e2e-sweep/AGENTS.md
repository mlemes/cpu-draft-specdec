# c3: end-to-end sweep

**Goal.** Measure end-to-end speedup of the CPU draft with GPU verification (H1).

**Settings.** Batch 1. Context {512, 2048}. γ {2, 4, 6}. Threads {3, 6}. Drafts: best D-S, best M, Qwen3-0.6B. Run a chosen subset, not the full grid: fix threads at the a2 optimum first, then vary γ and context.

**Optional variant.** The CPU drafts the next block while the GPU verifies the current one.

**Outputs.** Per configuration: tokens/s, time to first token, p50 and p95 latency, mean accepted length, round-time split (draft, verify, transfer, sync), peak VRAM and MBM bytes per token. Speedups as paired per-prompt ratios with bootstrap 95% CIs.

**Decision.** H1 holds if a draft reaches ≥ 1.3× with a CI lower bound > 1.0 and shows the CAT dependence from b5.
