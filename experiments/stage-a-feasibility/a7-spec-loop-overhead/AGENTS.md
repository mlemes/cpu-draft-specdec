# a7: speculative-loop overhead with a CPU draft

Added 2026-11-21 after a6: CPU-draft cycles were slower than GPU-draft cycles by more than γ·t_d. Goal: decompose one llama-server speculative cycle (draft-simple) into components and name where CPU-draft time goes. Method: `configs/probe.py` (llama-server flag variants, 10 general prompts, greedy, γ = 4; cycle time = predicted_ms / (draft_n / γ)) plus llama-bench component costs at depth 350. Results in `results/`.
