#!/usr/bin/env bash
# b7: continue b5 for 24M tokens with forward KL (control) and with the total-variation loss.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
PY=$HDD/venv-train/bin/python
TAG=$(sha256sum "$HDD/gguf/Qwen3-8B-Q4_K_M.gguf" | cut -c1-12)
T=$HDD/teacher
DATA="$T/general/$TAG:0.135,$T/general2/$TAG:0.225,$T/pt_general/$TAG:0.135,$T/pt_general2/$TAG:0.225,$T/pt_fin/$TAG:0.05,$T/json/$TAG:0.08,$T/json_extra/$TAG:0.15"
INIT=$HDD/runs/b2/kd-pruned-prose45json23-20261122T105044Z/final.pt
BATCH=${BATCH:-4}; ACCUM=${ACCUM:-4}
export PYTHONPATH=$R/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for loss in ${@:-kd tv}; do
  "$PY" "$R/src/train/train.py" --init "$INIT" --loss "$loss" --out "$HDD/runs/b2/b7-$loss-$(date -u +%Y%m%dT%H%M%SZ)" \
    --data "$DATA" --tokens 24000000 --batch-seqs "$BATCH" --grad-accum "$ACCUM" --seq-len 1024 --lr 1e-4 --warmup 100 \
    --eval-dir "$HDD/eval/$TAG" --eval-every-tokens 4000000 --ckpt-every-tokens 24000000 --seed 1 --device cuda
done
