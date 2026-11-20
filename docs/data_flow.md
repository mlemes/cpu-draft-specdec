# Data flow

Status: planned. Update this file whenever an artifact's format or location changes.

Large artifacts live under `/mnt/hdd1/cpu-draft-specdec/` (abbreviated `$HDD` below). Only small results and manifests are committed.

## Pipeline

1. **Corpora** (public only): general text, Portuguese financial text, structured JSON tasks. Raw files in `$HDD/corpora/<name>/`, with a manifest listing source URL, license, download date and SHA-256.
2. **Tokenization** with the target tokenizer. Output: token-ID shards in `$HDD/tokens/<corpus>/`.
3. **Vocabulary subset V′.** Token frequencies from the shards, then the target's probability mass on held-out text decides the sizes (experiment a5). Output: `$HDD/vocab/v_prime_<size>.json` with the ordered token IDs and coverage statistics.
4. **Teacher scoring** by the deployed Q4_K_M target with llama.cpp prefill (experiment b1). Output per token: top-32 token IDs (uint32), fp16 logits and the tail mass. About 194 B per token, in `$HDD/teacher/<corpus>/<target_sha>/`.
5. **On-policy data.** The student generates continuations, then the teacher scores them as in step 4.
6. **Training** (experiments b2–b4). Checkpoints in `$HDD/checkpoints/<student>/<run_id>/`.
7. **Export.** Hugging Face to GGUF, then quantization and the pruned head. Output: `$HDD/gguf/<student>/<run_id>-<quant>.gguf` with its SHA-256 recorded in the run manifest.
8. **Benchmarks and end-to-end runs** (Stages A–D). Per run: `experiments/<stage>/<exp>/results/<run_id>/manifest.json` and `metrics.jsonl` in the repository; raw logs, counter dumps and routing traces in `$HDD/runs/<run_id>/`.
9. **Online adaptation** (Stage D). Target distributions captured during verification go to `$HDD/teacher/online/<stream>/` in the format of step 4 and feed step 6.

## Retention

- Teacher data and GGUF files are kept until the thesis is submitted, because results must be reproducible from them.
- Random-weight GGUFs used for latency sweeps can be regenerated from their seed and configuration and may be deleted.
