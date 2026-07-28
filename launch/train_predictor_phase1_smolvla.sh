#!/usr/bin/env bash
# Phase 1 -- reconstruction: mse + (optional) feature. No policy loss.
# Set FEATURE_WEIGHT=0 to train on MSE alone; the released checkpoints used 1.0.
# Usage: launch/train_predictor_phase1_smolvla.sh <gpu> [output_dir]
set -euo pipefail
GPU=${1:?usage: train_predictor_phase1_smolvla.sh <gpu> [output_dir]}
OUT=${2:-outputs/predictor_smolvla_phase1}
REL=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
BANK=${BANK_DIR:-outputs/latent_bank_smolvla}
SIDECAR=${SIDECAR:-$BANK/state_task_sidecar.pt}
CORRECTOR=${CORRECTOR_CKPT:-weights/corrector.pt}
FEATURE_WEIGHT=${FEATURE_WEIGHT:-1.0}

cd "$REL"
mkdir -p "$OUT"
# a SHORT TMPDIR is required: DataLoader workers open a unix socket whose path is capped near
# 108 characters, and a long scratch path overflows it.
CUDA_VISIBLE_DEVICES=$GPU TMPDIR=${TMPDIR:-/tmp} \
"$PY" predictor/train.py \
  --backbone smolvla --data-dir "$BANK" --state-sidecar "$SIDECAR" \
  --corrector-checkpoint "$CORRECTOR" --output-dir "$OUT" \
  --steps 150000 --batch-size 256 --lr 3e-4 --lr-schedule cosine --lr-min 1e-6 \
  --mse-weight 1.0 --feature-weight "$FEATURE_WEIGHT" --policy-weight 0 \
  --max-delay 20 --num-workers 8 --seed 0 --ckpt-every 10000 --device cuda
