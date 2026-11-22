#!/usr/bin/env bash
# b5: does more distinct prose (instead of repeated passes) keep raising acceptance?
# One change against the 96M-token kd-pruned run: general and pt_general each get 20M new teacher-scored tokens
# (general2, pt_general2: documents disjoint from the b1 corpus, same sources and seed order). Prose share stays 0.87,
# split by token count; pt_fin 0.05 and json 0.08 unchanged. Same tokens, seed, init, schedule.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
M=$HDD/gguf/Qwen3-8B-Q4_K_M.gguf
TAG=$(sha256sum "$M" | cut -c1-12)
T=$HDD/teacher
for d in general2 pt_general2; do
  [ -f "$T/$d/$TAG/index.json" ] && grep -q '"complete": true' "$T/$d/$TAG/index.json" || \
    "$R/src/teacher/teacher_score" --model "$M" --input "$HDD/corpora/train/$d.jsonl" \
      --out "$T/$d/$TAG" --seq-len 1024 --k 32 --max-tokens 20000000 --shard-seqs 1024 --threads 5
done
DATA="$T/general/$TAG:0.163,$T/general2/$TAG:0.272,$T/pt_general/$TAG:0.163,$T/pt_general2/$TAG:0.272,$T/pt_fin/$TAG:0.05,$T/json/$TAG:0.08" \
RUN_TAG=prose44 TOKENS=96000000 "$R/experiments/stage-b-distillation/b2-dense-student/configs/run_b2_pilot.sh" kd-pruned
