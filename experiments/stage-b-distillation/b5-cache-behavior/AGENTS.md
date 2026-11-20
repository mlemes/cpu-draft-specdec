# b5: cache behavior

**Goal.** Measure whether the students' active weights stay resident in L3, and whether routing reuse explains the traffic (H3).

**Method.**
- For D-S, M (both arms) and D-L: run fixed prompt sets with CAT ways {1, 2, 4, 8, 16} at context 256 and 2048.
- Record t_d, MBM bytes per token, `llc_occupancy`, and core fill counters (only those a0 validated).
- From M's routing traces compute per-layer Jaccard similarity of expert sets between consecutive tokens, block unions at γ = 2, 4, 6, and reuse-distance histograms in bytes.
- Predict DRAM bytes per token from the reuse distances (stack-distance model) for each cache size, and compare with MBM.

**Outputs.** Measured versus predicted DRAM bytes per token, knee locations, and the effect of sticky routing.

**Confounds.** Context length grows the KV cache, which competes with weights; report KV bytes separately. Hardware prefetching and the L3 replacement policy are not exact LRU, so the stack-distance model is an approximation; report its error.
