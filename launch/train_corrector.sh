#!/usr/bin/env bash
# Train the full-state residual corrector on LIBERO demonstrations (no policy, no images).
# Usage: launch/train_corrector.sh <gpu> [output.pt]
set -euo pipefail
GPU=${1:?usage: train_corrector.sh <gpu> [output.pt]}
OUT=${2:-outputs/corrector/full_state_residual_d0_20.pt}
REL=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT to the local LeRobot LIBERO dataset}
DATASET_REVISION=${DATASET_REVISION:-86958911c0f959db2bbbdb107eb3e17c5f9c798e}

cd "$REL"
CUDA_VISIBLE_DEVICES=$GPU "$PY" corrector/train.py \
  --dataset-root "$DATASET_ROOT" --dataset-revision "$DATASET_REVISION" \
  --d-max 20 --steps 100000 --batch-size 512 \
  --hidden-dim 128 --num-layers 2 \
  --lr 1e-3 --lr-schedule cosine --lr-min 0 --weight-decay 1e-5 \
  --num-workers 8 --seed 1000 --device cuda \
  --output "$OUT"
