#!/usr/bin/env bash
# Build the per-frame visual-latent bank for pi05.
# Usage: launch/collect_pi05.sh <gpu> [worker_id] [num_workers]
# Run one process per worker_id, each on its own GPU, all writing to the same OUTPUT_DIR.
set -euo pipefail
GPU=${1:?usage: collect_pi05.sh <gpu> [worker_id] [num_workers]}
WORKER_ID=${2:-0}
NUM_WORKERS=${3:-1}
REL=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
POLICY=${POLICY_PATH:?set POLICY_PATH to the pi05 LIBERO checkpoint}
DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT to the local LeRobot LIBERO dataset}
DATASET_REVISION=${DATASET_REVISION:-86958911c0f959db2bbbdb107eb3e17c5f9c798e}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/latent_bank_pi05}
TOKENIZER=${TOKENIZER_PATH:-}

cd "$REL"
CACHE=$OUTPUT_DIR/_cache/w$WORKER_ID
mkdir -p "$CACHE/tmp" "$CACHE/ind" "$CACHE/tri"
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
TMPDIR=$CACHE/tmp TORCHINDUCTOR_CACHE_DIR=$CACHE/ind TRITON_CACHE_DIR=$CACHE/tri \
"$PY" predictor/collect_latents.py \
  --backbone pi05 --policy-path "$POLICY" \
  --dataset-root "$DATASET_ROOT" --dataset-revision "$DATASET_REVISION" \
  --output-dir "$OUTPUT_DIR" \
  ${TOKENIZER:+--tokenizer-path "$TOKENIZER"} \
  --dtype bfloat16 --batch-size 16 --shard-frames 20000 \
  --num-workers "$NUM_WORKERS" --worker-id "$WORKER_ID" --shard-index-stride 100000 \
  --device cuda
