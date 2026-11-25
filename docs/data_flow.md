# Data flow

Status (2026-11-21): steps 1, 3, 4 and 6 implemented as described; the others are planned. Update this file whenever an artifact's format or location changes.

Large artifacts live under `/mnt/hdd1/cpu-draft-specdec/` (abbreviated `$HDD` below). Only small results and manifests are committed.

## Pipeline

1. **Corpora** (public only): general text, Portuguese financial text, structured JSON tasks. Raw files in `$HDD/corpora/<name>/`, with a manifest listing source URL, license, download date and SHA-256.
2. **Training text** (`src/data/build_train_corpus.py`): `$HDD/corpora/train/<domain>.jsonl` (domains general, pt_general, pt_fin, json; evaluation material excluded) with `manifest.json`. Tokenization happens inside the teacher scorer.
3. **Vocabulary subset V′.** a5 ranks token IDs by frequency in domain text (`experiments/stage-a-feasibility/a5-vocab-coverage/results/<run>/ranking.json`); `src/train/vprime.py` takes the first 32,768 and forces the two Qwen3 end tokens in.
4. **Teacher scoring** (`src/teacher/teacher_score`, experiment b1): sequences of 1024 tokens; per position top-32 token IDs (uint32) and fp16 log-probabilities over the full vocabulary (tail mass = 1 − Σ p). 196 B per token, in `$HDD/teacher/<domain>/<target_sha12>/` with `index.json`.
4b. **Offline-acceptance evaluation set** (`src/data/gen_eval_set.py`): target greedy continuations of the a4 prompts with per-position top-32 log-probabilities, `$HDD/eval/<target_sha12>/eval_<domain>.jsonl`.
5. **On-policy data.** The student generates continuations, then the teacher scores them as in step 4.
6. **Training** (experiments b2–b4, `src/train/train.py`). Run directories `$HDD/runs/b2/<run_id>/` with `metrics.jsonl`, checkpoints (`config`, float32 `state_dict`, `vprime`) and `manifest.json`; small summaries are copied to `experiments/stage-b-distillation/<exp>/results/<run_id>/`. For b10 the per-arm init checkpoints are written by `train.ffn_init` to `$HDD/runs/b10/init/<arm>.pt` and the training runs to `$HDD/runs/b10/<run_id>/` (spec: `docs/specs/b10-ffn-variants.md`).
7. **Export** (`src/models/export_student_gguf.py`). Student `.pt` checkpoint to Qwen3 GGUF with `output_vmap.weight` and copied tokenizer keys, then quantization. Output: `$HDD/gguf/<student>/<run_id>-<quant>.gguf` with its SHA-256 recorded in the run manifest.
8. **Benchmarks and end-to-end runs** (Stages A–D). Per run: `experiments/<stage>/<exp>/results/<run_id>/manifest.json` and `metrics.jsonl` in the repository; raw logs, counter dumps and routing traces in `$HDD/runs/<run_id>/`.
9. **Online adaptation** (Stage D). Target distributions captured during verification go to `$HDD/teacher/online/<stream>/` in the format of step 4 and feed step 6.

## Retention

- Teacher data and GGUF files are kept until the thesis is submitted, because results must be reproducible from them.
- Random-weight GGUFs used for latency sweeps can be regenerated from their seed and configuration and may be deleted.
