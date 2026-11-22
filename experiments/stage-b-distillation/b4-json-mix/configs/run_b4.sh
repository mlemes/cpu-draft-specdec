#!/usr/bin/env bash
# b4: does more structured-output data raise JSON acceptance without hurting prose?
# One change against the 96M-token kd-pruned run: the data mix adds json_extra (Hermes function-calling and
# agentic JSON, ~10M tokens), JSON share 0.08 -> 0.23, prose shares scaled down. Same tokens, seed, init, schedule.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
M=$HDD/gguf/Qwen3-8B-Q4_K_M.gguf
TAG=$(sha256sum "$M" | cut -c1-12)
T=$HDD/teacher
[ -f "$T/json_extra/$TAG/index.json" ] && grep -q '"complete": true' "$T/json_extra/$TAG/index.json" || \
  "$R/src/teacher/teacher_score" --model "$M" --input "$HDD/corpora/train/json_extra.jsonl" \
    --out "$T/json_extra/$TAG" --seq-len 1024 --k 32 --max-tokens 100000000 --shard-seqs 1024 --threads 5 --parse-special
DATA="$T/general/$TAG:0.36,$T/pt_general/$TAG:0.36,$T/pt_fin/$TAG:0.05,$T/json/$TAG:0.08,$T/json_extra/$TAG:0.15" \
RUN_TAG=json23 TOKENS=96000000 "$R/experiments/stage-b-distillation/b2-dense-student/configs/run_b2_pilot.sh" kd-pruned
