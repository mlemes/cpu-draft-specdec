# b4: dense control D-L

**Goal.** Separate "more stored parameters" from "conditional computation". D-L is a dense model with about the same stored bytes as M (≈ 150–160 MB at Q8_0), so all of its weights are touched every token.

**Method.** Same data, objective (b2 arm 2) and token budget as M. Pick hidden size and depth to match M's stored bytes; record the exact configuration.

**Outputs.** Acceptance, t_d and MBM bytes per token. Expected to be DRAM-bound; its role is to show what M's extra parameters are worth without routing.
