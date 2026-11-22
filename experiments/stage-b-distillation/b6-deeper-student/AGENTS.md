# b6: a deeper student within the CPU-cache budget

Question: b5 showed that new prose does not lift the prose plateau (general 0.57 against 0.63 for Qwen3-0.6B), so is
student capacity the limit? And does the extra agreement pay for the slower drafting?

Single controlled change against b5 (same data mix, tokens, seed, schedule, eval): 18 layers instead of 12, pruned from
Qwen3-0.6B with evenly spaced source layers (`src/train/prune_init.py --config '{"n_layers": 18}'`). Width, heads and
V′ unchanged. Per-token weights read: about 71M parameters (about 75 MB at Q8_0) against 53M, still under the 96 MB L3.
Draft time should rise by about a third.

Run: `configs/run_b6.sh`. Compare offline greedy agreement per domain at 96M tokens against b5; if it gains, export and
benchmark as a draft against the b4/b5-mix 12-layer student.
