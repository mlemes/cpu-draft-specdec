#!/usr/bin/env bash
# b6: 18-layer kd-pruned student on the b5 mix (one change against b5: depth). Same settings as run_b2_pilot.sh.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
PY=$HDD/venv-train/bin/python
TAG=$(sha256sum "$HDD/gguf/Qwen3-8B-Q4_K_M.gguf" | cut -c1-12)
T=$HDD/teacher
DATA="$T/general/$TAG:0.135,$T/general2/$TAG:0.225,$T/pt_general/$TAG:0.135,$T/pt_general2/$TAG:0.225,$T/pt_fin/$TAG:0.05,$T/json/$TAG:0.08,$T/json_extra/$TAG:0.15"
INIT=$HDD/runs/b2/init/pruned-qwen3-0.6b-L18.pt
BATCH=${BATCH:-4}; ACCUM=${ACCUM:-4}  # effective batch 16 sequences; use 2 x 8 if 4 x 4 runs out of memory
export PYTHONPATH=$R/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$INIT" ] || "$PY" "$R/src/train/prune_init.py" --src "$HDD/hf/Qwen3-0.6B" --config '{"n_layers": 18}' \
  --vprime-ranking "$R/experiments/stage-a-feasibility/a5-vocab-coverage/results/a5-20261121T011928Z/ranking.json" --out "$INIT"
"$PY" "$R/src/train/train.py" --init "$INIT" --loss kd --out "$HDD/runs/b2/kd-pruned-L18-$(date -u +%Y%m%dT%H%M%SZ)" \
  --data "$DATA" --tokens 96000000 --batch-seqs "$BATCH" --grad-accum "$ACCUM" --seq-len 1024 --lr 6e-4 --warmup 100 \
  --eval-dir "$HDD/eval/$TAG" --eval-every-tokens 4000000 --ckpt-every-tokens 8000000 --seed 0 --device cuda
