# b5: more distinct prose

Question: the 96M-token kd-pruned run flattened after about 60M tokens while repeating 24.6M scored tokens. Does new
text keep acceptance rising on general and pt_fin prose?

Single controlled change against b4 (same 96M tokens, JSON share 0.23, which matched the baseline on prose and beat it on JSON): 20M new teacher-scored tokens each for English and Portuguese Wikipedia
(`general2`, `pt_general2`, documents disjoint from the b1 corpus; built by `src/data/build_train_corpus.py` with
200M-character budgets, keeping only documents after the b1 prefix). Mix weights keep the prose share at 0.72 (old:new 0.375:0.625 by token count) and b4's JSON weights.

Run: `configs/run_b5.sh`. Compare offline greedy agreement per domain at 96M tokens against b4.
