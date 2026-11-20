# b6: standalone student quality

**Goal.** Report each student's quality as a model on its own, separately from speculative decoding.

**Metrics.**
- Perplexity on held-out text per domain. Restricted-head students are scored over V′, so also report the out-of-V′ token rate.
- Top-1 agreement and TV distance with the teacher.
- A small task set (Portuguese reading and financial questions, JSON validity).
- CPU latency per token, giving the quality–latency tradeoff.

**Outputs.** One table per student. Include Qwen3-0.6B as the reference.
