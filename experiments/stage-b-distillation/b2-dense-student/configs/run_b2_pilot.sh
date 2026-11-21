#!/usr/bin/env bash
# b2 pilot ("short Stage B check"): D-S on the b1 pilot teacher data (~25M tokens), three arms.
#   kd-pruned  : forward KL on top-32, init pruned from Qwen3-0.6B   (main arm)
#   hard-pruned: hard labels only, same init                         (does KD help?)
#   kd-scratch : forward KL, random init                             (does pruning help?)
# Usage: run_b2_pilot.sh [arm ...]   (default: all three, sequentially; GPU must be otherwise idle)
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
PY=$HDD/venv-train/bin/python
TAG=$(sha256sum "$HDD/gguf/Qwen3-8B-Q4_K_M.gguf" | cut -c1-12)
T=$HDD/teacher
DATA="$T/general/$TAG:0.435,$T/pt_general/$TAG:0.435,$T/pt_fin/$TAG:0.05,$T/json/$TAG:0.08"
EVAL=$HDD/eval/$TAG
INIT=$HDD/runs/b2/init/pruned-qwen3-0.6b.pt
TOKENS=${TOKENS:-24000000}
export PYTHONPATH=$R/src
mkdir -p "$HDD/runs/b2/init"
[ -f "$INIT" ] || "$PY" "$R/src/train/prune_init.py" --src "$HDD/hf/Qwen3-0.6B" \
  --vprime-ranking "$R/experiments/stage-a-feasibility/a5-vocab-coverage/results/a5-20261121T011928Z/ranking.json" --out "$INIT"
common=(--data "$DATA" --tokens "$TOKENS" --batch-seqs 8 --grad-accum 2 --seq-len 1024 --lr 6e-4 --warmup 100
        --eval-dir "$EVAL" --eval-every-tokens 4000000 --ckpt-every-tokens 8000000 --seed 0 --device cuda)
for arm in "${@:-kd-pruned hard-pruned kd-scratch}"; do
  for a in $arm; do
    out=$HDD/runs/b2/$a-$(date -u +%Y%m%dT%H%M%SZ)
    case $a in
      kd-pruned)   "$PY" "$R/src/train/train.py" --init "$INIT" --loss kd   --out "$out" "${common[@]}" ;;
      hard-pruned) "$PY" "$R/src/train/train.py" --init "$INIT" --loss hard --out "$out" "${common[@]}" ;;
      kd-scratch)  "$PY" "$R/src/train/train.py" --init scratch --loss kd   --out "$out" "${common[@]}" ;;
      *) echo "unknown arm $a" >&2; exit 2 ;;
    esac
  done
done
