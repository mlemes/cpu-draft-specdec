#!/usr/bin/env bash
# b9: per-pass target time (GPU) and per-token draft time (CPU, 4 threads) at 512-token depth.
set -euo pipefail
B=third_party/llama.cpp/build/bin; G=/mnt/hdd1/cpu-draft-specdec/gguf
OUT=${1:-experiments/stage-b-distillation/b9-overlap-bound/results/b9-20261122}
$B/llama-bench -m $G/Qwen3-8B-Q4_K_M.gguf -ngl 99 -p 1,2,3,4,5,7 -n 64 -d 512 -r 5 -o csv > $OUT/tgt.csv
$B/llama-bench -m $G/students/ds-kd-pruned-json23-96M-Q8_0.gguf -ngl 0 -dev none -t 4 -p 1,2,3,5 -n 64 -d 512 -r 5 -o csv > $OUT/dft.csv
