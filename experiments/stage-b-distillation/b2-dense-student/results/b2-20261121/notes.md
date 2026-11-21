# b2 pilot: D-S student, three arms, 24M tokens each (2026-11-21)

Student: D-S (`StudentConfig` defaults, V' = 32,768 head). Teacher data: b1 pilot (`../../b1-teacher-data/results/b1-20261121`).
Script: `configs/run_b2_pilot.sh` at commit c31238f. Batch 4 x 1024 x grad-accum 4 (16,384 tokens/step), lr 6e-4,
warmup 100, data weights general 0.435, pt_general 0.435, pt_fin 0.05, json 0.08. RTX 3070, bf16 autocast,
about 29k tokens/s (14 min per arm). Runs: `/mnt/hdd1/cpu-draft-specdec/runs/b2/<arm>-20261121T*`.

Eval set: `/mnt/hdd1/cpu-draft-specdec/eval/d98cdcbd03e1`, Qwen3-8B-Q4_K_M greedy continuations of the a4 prompts
(100 per domain, 128 tokens max; 12,737 / 12,698 / 7,861 positions). Two rows were truncated at a server row/token
mismatch (general-078 to 65 tokens, pt_fin-038 to 26). Metric: greedy agreement = P(student argmax = target token),
the per-position acceptance rate of greedy speculative decoding; 95% CI from the prompt-level bootstrap.

## Greedy agreement at 24M tokens

| arm | general | pt_fin | json | final train loss |
|---|---|---|---|---|
| kd-pruned | **0.438** [0.421, 0.456] | **0.590** [0.577, 0.602] | **0.527** [0.509, 0.545] | 1.94 (KD) |
| hard-pruned | 0.403 [0.385, 0.421] | 0.477 [0.464, 0.490] | 0.466 [0.446, 0.487] | 3.74 (CE) |
| kd-scratch | 0.407 [0.388, 0.425] | 0.576 [0.563, 0.589] | 0.425 [0.403, 0.446] | 2.36 (KD) |

Trajectory for kd-pruned (general / pt_fin / json): 4M 0.30/0.44/0.28, 12M 0.38/0.54/0.40, 24M 0.44/0.59/0.53.
Gains per 4M tokens were still +0.01 to +0.03 at the end with the learning rate decayed to 10%; the pilot is not converged.

## Reading

- KD beats hard labels in every domain; the gap is largest on pt_fin (+0.11) and json (+0.06).
- Pruned init beats scratch on general (+0.03) and json (+0.10); on pt_fin the intervals overlap.
  The CIs above are per arm; paired comparisons on the same prompts would be tighter and were not computed.
- Against the Stage A cost model: kd-pruned clears the ~0.40 break-even in all three domains and the ~0.55
  needed for 1.3x only on pt_fin. Qwen3-0.6B reaches 0.63-0.66 in a4; D-S has about 53M non-embedding parameters (including the V' head) against about 440M, roughly one eighth.
- pt_fin training text (Copom 232-261, 136k tokens) repeats about 9 times at weight 0.05; evaluation uses held-out
  minutes 262-282, so the pt_fin score is not memorization of the eval text but may reflect overfitting to its style.

## Next

1. Longer kd-pruned run (more tokens; the teacher corpus has 24.6M scored tokens, so either score more or repeat).
2. GGUF export with the V' head and a real speculative run in llama.cpp to measure end-to-end speedup.
