# CPU draft, GPU target: cache-resident speculative decoding

Master's research project. It asks whether a small model distilled from a GPU target model, running on the CPU, can speed up the target's generation through speculative decoding. The idea is that the draft's active weights stay resident in the CPU's 96 MB L3 cache. The full design is in `docs/proposal.md`; read it before working on any experiment.

## Objectives

1. Measure whether a distilled CPU draft speeds up a GPU target end to end at batch 1, counting drafting, verification, PCIe transfer and synchronization.
2. At a fixed active-weight budget, compare a conditional (MoE) student with a small dense student.
3. Test whether temporal reuse of active weights produces measurable cache benefits, with causal evidence from L3 allocation control (CAT) and DRAM traffic counters (MBM).
4. Find when refreshing the student under changing workloads pays back its cost.
5. Report the student's standalone quality separately from speculative decoding.

These are hypotheses. A well-measured negative result is a valid outcome, and each stage has a stop gate.

## Fixed setup

- **Machine:** Ryzen 5 5500X3D (6 cores, 96 MiB 16-way L3, AVX2, no AVX-512 or VNNI), 30 GiB RAM, RTX 3070 8 GiB on PCIe 3.0 x16, Ubuntu 26.04, Linux 7.0. Details in `docs/proposal.md` section 3.
- **Target:** Qwen3-8B Q4_K_M on the GPU (fallback Qwen3-4B Q4_K_M). Once chosen, the target file, quantization and sampler settings never change between compared runs.
- **Students:** dense D-S (about 56 MB touched per token at Q8_0), MoE M (same active bytes, about 162 MB stored), dense control D-L (about 160 MB). Every student keeps the target's tokenizer and token IDs; the output head covers a pruned subset V′ of the vocabulary.
- **Runtime:** llama.cpp in one process, target on CUDA and draft on CPU, plus a small patch for the pruned draft head.

## Layout

| Path | Contents |
|---|---|
| `docs/proposal.md` | Full research proposal: hypotheses, budgets, correctness, stage plan, sources |
| `docs/architecture.md` | Components and how they connect |
| `docs/data_flow.md` | Data and artifacts from teacher scoring to results |
| `experiments/` | One folder per stage, one subfolder per experiment, each with its own `AGENTS.md`, `configs/` and `results/` |
| `src/` | Shared code: speculative decoding loop, measurement, training, model building, llama.cpp patches |
| `data/` | Local pointers only; large data lives on `/mnt/hdd1/cpu-draft-specdec/` |
| `third_party/` | Pinned llama.cpp checkout (git-ignored) |

Stages run in order, and each stage's gate decides whether the next one starts: A (feasibility) → B (distillation and cache) → C (end-to-end) → D (adaptation). Experiment `a0-counter-validation` must pass before any cache claim is made.

## Rules for agents

1. **Label every factual claim**: [V] verified, [R] cited but not opened, [E] estimate with its assumptions, [U] unknown. Never promote [R] or [E] to [V] without evidence.
2. **Bytes, not parameters.** Memory claims use the runtime representation after repacking: stored bytes, touched bytes per token, block union, and non-weight state, each reported separately (proposal section 4). A checkpoint size is not evidence of cache residency.
3. **One change at a time.** Compared runs differ in one controlled variable only. Record the others in the manifest.
4. **Benchmark hygiene:** SMT off, governor `performance`, threads pinned to physical cores, other processes confined to disjoint CAT ways, VRAM freed, swap activity checked. Report prefill and decode separately, and cold and warm runs separately. Use at least 5 repeats, fixed seeds, and paired bootstrap 95% confidence intervals.
5. **Counters:** trust a counter only after `a0-counter-validation` passes. Name its limitation next to every number it produces.
6. **Root steps are the operator's job.** `perf_event_paranoid`, `modprobe amd_uncore`, mounting resctrl and toggling SMT need sudo. Ask the operator; do not work around missing privileges.
7. **Large files stay out of the repository:** models, teacher logits, checkpoints and raw logs go under `/mnt/hdd1/cpu-draft-specdec/`. `/` has only about 38 GB free.
8. **Data:** public data only. No bank or customer data.
9. **Do not edit `docs/proposal.md` silently.** Record design changes in it with a dated note, and update `docs/architecture.md` and `docs/data_flow.md` when behavior changes.

## Run manifest

Every run writes `results/<run_id>/manifest.json` with:

- run ID, UTC timestamp, git commit of this project, llama.cpp commit and build flags;
- SHA-256 of every model file, quantization types, and touched/stored byte counts;
- kernel version, SMT state, governor, thread count and CPU affinity, CAT bit mask, MBA setting;
- free VRAM at start and other GPU clients;
- context length, γ, sampler chain and settings, seed;
- prompt-set ID and hash.

Metrics go in `results/<run_id>/metrics.jsonl`. Raw logs go to `/mnt/hdd1/cpu-draft-specdec/runs/<run_id>/`.
