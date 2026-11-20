# c1: decoding correctness

**Goal.** Show that speculative decoding preserves the target distribution it claims to preserve.

**Tests.**
1. Exact enumeration of the accept, residual and bonus steps on a toy vocabulary with known p and q (unit test in `src/specdec/`).
2. Greedy: compare outputs with target-only greedy decoding on the fixed prompt sets. Report the argmax flip rate and KL between the verify-path and decode-path target distributions; the preserved distribution is the verify path's.
3. Stochastic: chi-square and TV tests on the distribution of the first k generated tokens, speculative versus non-speculative, with fixed prompts and many seeds.
4. Sampler chain: temperature, top-k, top-p, min-p and penalties applied identically to p̃ and q̃.
5. KV rollback: after each round both caches hold exactly the committed prefix.
6. Audit llama.cpp's `common_sampler_sample_and_accept_n_rejection` before relying on it.

**Pass.** All tests pass; any remaining numerical difference is quantified and stated.
