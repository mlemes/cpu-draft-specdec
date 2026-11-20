# a2: CPU draft latency and cache sweep

**Goal.** Measure draft time per token t_d as a function of touched bytes, precision, thread count and L3 allocation, and find the cache knee.

**Inputs.** Random-weight GGUFs in the D-S layout (`src/models/`), scaled so touched bytes span 16–512 MB, in Q4_0 and Q8_0. Latency does not depend on weight values for dense models. For MoE layouts, fix the routing with a recorded trace so the experts touched are controlled.

**Method.**
- Grid: touched bytes {16, 32, 48, 64, 96, 128, 256, 512} MB × {Q4_0, Q8_0} × threads {1, 3, 6} with SMT off.
- At the sizes around 96 MB, sweep CAT ways {1, 2, 4, 8, 16}.
- Counters as validated in a0. Context fixed at 256 tokens.
- Record repacked heap bytes reported by the runtime, not only file size.

**Outputs.** t_d curves per byte and per FLOP; MBM bytes per token; the knee location with CI.

**Stop signal for Stage A.** t_d ≥ 5 ms at ≤ 60 MB, or no knee in latency and MBM bytes per token.

**Confounds.** A size sweep changes compute as well as bytes, so the CAT sweep is the causal evidence. Q4 and Q8 differ in dequantization work as well as bytes.
