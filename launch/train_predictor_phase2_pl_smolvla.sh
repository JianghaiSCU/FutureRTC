#!/usr/bin/env bash
# Phase 2 -- policy distillation on top of a phase-1 checkpoint.
# Usage: launch/train_predictor_phase2_pl_smolvla.sh <gpu> <phase1_checkpoint> [output_dir]
set -euo pipefail
GPU=${1:?usage: train_predictor_phase2_pl_smolvla.sh <gpu> <phase1_ckpt> [output_dir]}
BASE=${2:?phase-1 checkpoint required}
OUT=${3:-outputs/predictor_smolvla_phase2_pl}
REL=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
BANK=${BANK_DIR:-outputs/latent_bank_smolvla}
SIDECAR=${SIDECAR:-$BANK/state_task_sidecar.pt}
CORRECTOR=${CORRECTOR_CKPT:-weights/corrector.pt}
POLICY=${POLICY_PATH:?set POLICY_PATH to the smolvla LIBERO checkpoint}
TOKENIZER=${TOKENIZER_PATH:-}

cd "$REL"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU TMPDIR=${TMPDIR:-/tmp} \
"$PY" predictor/train.py \
  --backbone smolvla --data-dir "$BANK" --state-sidecar "$SIDECAR" --sidecar "$SIDECAR" \
  --corrector-checkpoint "$CORRECTOR" --output-dir "$OUT" \
  --resume-from "$BASE" --start-step 0 \
  --steps 100000 --batch-size 32 --lr 1e-4 --lr-schedule cosine --lr-min 1e-6 \
  --mse-weight 1.0 --feature-weight 1.0 --policy-weight 10.0 \
  --policy-path "$POLICY" ${TOKENIZER:+--tokenizer-path "$TOKENIZER"} \
  --max-delay 20 --num-workers 8 --seed 0 --ckpt-every 10000 --device cuda
