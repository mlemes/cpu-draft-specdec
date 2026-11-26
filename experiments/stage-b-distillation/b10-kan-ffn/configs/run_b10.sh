#!/usr/bin/env bash
# b10: FFN variants at the D-S touched-byte budget (one change per arm; see ../AGENTS.md).
# Usage: configs/run_b10.sh <arm> [...]   arms: ds-cont ds-fresh mlp-silu mlp-rational gr-kan kan-bspline kan-bspline-fit
# Same variables, data mix and training flags as ../../b6-deeper-student/configs/run_b6.sh, plus --compile
# (bspline_kan: 21.1k against 4.7k tokens/s; loss after 40 steps 4.551 against 4.554) and checkpoints every 32M tokens.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
HDD=/mnt/hdd1/cpu-draft-specdec
PY=$HDD/venv-train/bin/python
TAG=$(sha256sum "$HDD/gguf/Qwen3-8B-Q4_K_M.gguf" | cut -c1-12)
T=$HDD/teacher
DATA="$T/general/$TAG:0.135,$T/general2/$TAG:0.225,$T/pt_general/$TAG:0.135,$T/pt_general2/$TAG:0.225,$T/pt_fin/$TAG:0.05,$T/json/$TAG:0.08,$T/json_extra/$TAG:0.15"
# b5 D-S checkpoint: source of every non-FFN weight and of the FFN in ds-cont.
DSREF=${DSREF:-$HDD/runs/b2/kd-pruned-prose45json23-20261122T105044Z/final.pt}
BATCH=${BATCH:-4}; ACCUM=${ACCUM:-4}  # effective batch 16 sequences, as in b6
SEED=${SEED:-0}
INITDIR=$HDD/runs/b10/init
# Fit of kan-bspline-fit: --fit-tokens 4000000 at 4 x 1024 tokens per batch = 977 batches,
# so --fit-steps 1954 is two epochs over that data. --fit-lr 3e-3 is an Adam step on a
# fixed per-layer regression (larger than the KD lr 6e-4, which also trains the rest of
# the model). These are a wall-clock/quality tradeoff, not tuned values [E].
FIT="--fit --fit-data $DATA --fit-tokens 4000000 --fit-steps 1954 --fit-lr 3e-3 --fit-batch-seqs 4 --device cuda"

[ $# -ge 1 ] || { echo "usage: $0 <arm> [...]" >&2; exit 2; }

export PYTHONPATH=$R/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ARM in "$@"; do
  case "$ARM" in
    ds-cont)          CONFIG=''; FITFLAGS=''; INIT=$DSREF ;;
    ds-fresh)         CONFIG='{}'; FITFLAGS='' ;;
    mlp-silu)         CONFIG='{"ffn_type": "mlp", "ffn": 2304}'; FITFLAGS='' ;;
    mlp-rational)     CONFIG='{"ffn_type": "mlp_rational", "ffn": 2304}'; FITFLAGS='' ;;
    gr-kan)           CONFIG='{"ffn_type": "gr_kan", "ffn": 2304}'; FITFLAGS='' ;;
    kan-bspline)      CONFIG='{"ffn_type": "bspline_kan", "ffn": 256}'; FITFLAGS='' ;;
    kan-bspline-fit)  CONFIG='{"ffn_type": "bspline_kan", "ffn": 256}'; FITFLAGS=$FIT ;;
    *) echo "unknown arm: $ARM (ds-ref is not retrained)" >&2; exit 2 ;;
  esac

  if [ -z "${INIT:-}" ]; then
    INIT=$INITDIR/$ARM.pt
    if [ ! -f "$INIT" ]; then
      mkdir -p "$INITDIR"
      "$PY" -m train.ffn_init --src "$DSREF" --config "$CONFIG" --out "$INIT" --seed "$SEED" $FITFLAGS
    fi
  fi

  "$PY" "$R/src/train/train.py" --init "$INIT" --loss kd --out "$HDD/runs/b10/$ARM-$(date -u +%Y%m%dT%H%M%SZ)" \
    --data "$DATA" --tokens 96000000 --batch-seqs "$BATCH" --grad-accum "$ACCUM" --seq-len 1024 --lr 6e-4 --warmup 100 \
    --eval-dir "$HDD/eval/$TAG" --eval-every-tokens 4000000 --ckpt-every-tokens 32000000 --seed "$SEED" --device cuda --compile
  unset INIT
done
