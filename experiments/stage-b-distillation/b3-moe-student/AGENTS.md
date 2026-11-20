# b3: MoE student M

**Goal.** Test whether conditional computation beats the dense student at the same active bytes per token (H2), and whether routing reuse can be shaped (H3).

**Architecture.** D-S attention; FFN = shared expert of width 768 plus 16 routed experts of width 384, top-2. Active about 56 MB at Q8_0, stored about 162 MB.

**Method.**
1. Upcycle from the D-S arm-2 checkpoint: split the FFN's 1536 neurons into four groups of 384; two form the shared expert, the other two are each replicated with noise into eight experts. Router near uniform.
2. Continue training for the same tokens as the D-S continuation control. Add a load-balancing loss.
3. Second arm: add a sticky-routing penalty on changes of the top-2 set between consecutive tokens. Sweep its weight at 2–3 values.

**Open item.** Confirm which llama.cpp MoE architecture supports a shared expert before fixing the tensor layout [U].

**Outputs.** Acceptance per domain, routing traces for b5, and GGUF exports with per-expert byte counts.
