# a5 run 2026-11-21T01:19:28Z: findings

Target Qwen3-8B Q4_K_M on GPU. V′ ranked by token frequency in the frequency texts of all three domains (WikiText-103 validation, Copom minutes 232–261, Hermes json-mode rows not used as prompts). Evaluation on prompt + held-out continuation of the 300 prompt-set rows (ctx 1024). 95% CIs: percentile bootstrap over rows; rows cut from the same document are correlated, so true intervals are wider.

| Domain | positions | mass in V′ 16k | mass in V′ 32k | mass in V′ 64k | greedy token in V′ 32k |
|---|---|---|---|---|---|
| general | 82,244 | 0.918 (0.913–0.923) | 0.960 (0.956–0.963) | see summary.json | 0.966 |
| pt_fin | 99,982 | 0.967 (0.966–0.969) | 0.984 (0.983–0.984) | see summary.json | 0.993 |
| json | 40,672 | 0.956 (0.953–0.960) | 0.981 (0.979–0.983) | see summary.json | 0.983 |

Gate: p(V′ = 32k) ≥ 0.9 passes in every domain [V]. With V′ = 32k the acceptance ceiling from head pruning alone is about 0.96–0.98. The frequency ranking is fitted on text from the same domains, so out-of-domain coverage may be lower [U].
