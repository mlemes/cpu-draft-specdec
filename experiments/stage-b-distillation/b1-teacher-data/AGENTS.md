# b1: teacher data

**Goal.** Produce distillation targets from the exact deployed target, Qwen3-8B Q4_K_M.

**Method.**
- Prefill scoring with llama.cpp on the token shards. Store per position: top-32 token IDs (uint32), fp16 logits, and fp16 tail mass (1 − top-32 mass). About 194 B per token.
- On-policy data: the current student generates continuations of training prompts; the teacher scores them the same way.
- Budget: about 100M tokens (≈ 19 GB, ≈ 14 h at an estimated 2k tokens/s). Measure the real rate first and update the estimate.

**Outputs.** `$HDD/teacher/<corpus>/<target_sha>/` shards with an index and checksums. Report throughput, total GPU hours and the tail-mass distribution.

**Checks.** Tail mass statistics per domain; spot-check 100 positions against full-vocabulary logits.
