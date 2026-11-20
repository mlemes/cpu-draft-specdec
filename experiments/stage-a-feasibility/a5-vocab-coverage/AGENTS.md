# a5: draft vocabulary coverage

**Goal.** Measure p(V′), the target's expected probability mass inside a pruned draft vocabulary. It upper-bounds acceptance: α ≤ p(V′).

**Method.**
1. Rank tokens by frequency on training shards of the three domains.
2. Build V′ of sizes 16k, 32k and 64k.
3. On held-out text from each domain, score with the deployed target and compute the mean mass inside V′, and the fraction of greedy target tokens inside V′.

**Outputs.** Coverage per size and domain with CI; `$HDD/vocab/v_prime_<size>.json`.

**Stop signal.** p(V′ = 32k) < 0.9: reconsider the head budget before Stage B.
