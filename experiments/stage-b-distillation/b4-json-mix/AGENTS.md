# b4: structured-output share in the distillation mix

Question: the 24M-token student reached 1.11x on JSON against 1.90x for Qwen3-0.6B (b3). Does a larger share of
structured-output data close that gap, and what does it cost on prose?

Single controlled change against the 96M-token kd-pruned run (`b2-dense-student`, run tag none): the data mix adds
`json_extra` (built by `src/data/build_json_extra.py` from four other Hermes function-calling files, Apache-2.0),
raising the JSON share from 0.08 to 0.23. Init, seed, tokens, schedule and eval set are identical.

Run: `configs/run_b4.sh` (scores json_extra with the teacher if needed, then trains). Compare offline greedy agreement
per domain at 96M tokens; the evaluation JSON prompts come from json-mode-singleturn, which json_extra excludes.
