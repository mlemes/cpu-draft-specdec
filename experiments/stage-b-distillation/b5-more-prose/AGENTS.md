# b5: more distinct prose

Question: the 96M-token kd-pruned run flattened after about 60M tokens while repeating 24.6M scored tokens. Does new
text keep acceptance rising on general and pt_fin prose?

Single controlled change against that run: 20M new teacher-scored tokens each for English and Portuguese Wikipedia
(`general2`, `pt_general2`, documents disjoint from the b1 corpus; built by `src/data/build_train_corpus.py` with
200M-character budgets, keeping only documents after the b1 prefix). Mix weights keep the prose share at 0.87.

Run: `configs/run_b5.sh`. Compare offline greedy agreement per domain at 96M tokens against the 96M baseline and b4.
