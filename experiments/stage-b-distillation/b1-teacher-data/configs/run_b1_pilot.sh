#!/usr/bin/env bash
# b1 pilot: score the Stage B training corpus with the deployed target (Qwen3-8B Q4_K_M).
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
M=$HDD/gguf/Qwen3-8B-Q4_K_M.gguf
TAG=$(sha256sum "$M" | cut -c1-12)
run() {  # domain max_tokens [extra]
  local d=$1 n=$2; shift 2
  "$R/src/teacher/teacher_score" --model "$M" --input "$HDD/corpora/train/$d.jsonl" \
    --out "$HDD/teacher/$d/$TAG" --seq-len 1024 --k 32 --max-tokens "$n" --shard-seqs 1024 --threads 5 "$@"
}
mkdir -p "$HDD/teacher"
run pt_fin 100000000
run json 100000000 --parse-special
run general 12000000
run pt_general 12000000
