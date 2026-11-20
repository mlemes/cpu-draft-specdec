# Shared code

Code used by more than one experiment. Experiment folders hold configurations and results, not code. See `../docs/architecture.md` for how the parts connect.

| Folder | Contents |
|---|---|
| `specdec/` | Speculative decoding loop on llama.cpp: CPU draft, GPU verification, greedy and stochastic acceptance, KV rollback, per-round timing |
| `measure/` | resctrl control (CAT masks, MBM, `llc_occupancy`), `perf` wrappers, IBS sampling, touched-byte calculator, run-manifest writer, bootstrap confidence intervals |
| `train/` | Distillation trainer: data loader for top-k teacher records, hard-label, forward-KL and TV losses, chunked head loss, MoE upcycling, sticky-routing penalty |
| `models/` | Student configurations (D-S, M, D-L), structured pruning of Qwen3-0.6B, GGUF export and quantization, random-weight GGUF generator |
| `patches/` | Patches against the pinned llama.cpp commit, for example the pruned draft head over V′ |

## Conventions

- Python through `uv`. Run scripts that read downloaded data with `python -I`.
- Every entry point writes the run manifest described in the root `AGENTS.md`.
- Speculative-decoding acceptance code needs a test that enumerates a toy vocabulary exactly before any end-to-end run uses it.
- Patches name the llama.cpp commit they apply to. Rebasing to a new commit is a recorded design change.
