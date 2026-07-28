#!/usr/bin/env bash
# Multi-delay LIBERO eval of predictor + corrector for pi05.
# All delays run concurrently in one env batch; suites run sequentially on one GPU.
# Usage: launch/eval_pi05.sh <gpu> <predictor_ckpt> <out_base> [suite:max_steps ...]
# Default suites: libero_spatial:220 libero_object:280 libero_goal:300 libero_10:520
set -euo pipefail
GPU=${1:?usage: eval_pi05.sh <gpu> <predictor_ckpt> <out_base> [suite:max_steps ...]}
CKPT=${2:?predictor checkpoint required}
OUTBASE=${3:?output base dir required}
shift 3
SUITES=("$@")
if [ ${#SUITES[@]} -eq 0 ]; then
  SUITES=(libero_spatial:220 libero_object:280 libero_goal:300 libero_10:520)
fi

REL=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
POLICY=${POLICY_PATH:?set POLICY_PATH to the pi05 LIBERO checkpoint}
CORRECTOR=${CORRECTOR_CKPT:-weights/corrector.pt}
TOKENIZER=${TOKENIZER_PATH:-}
DELAYS=${DELAYS:-5,10,15,20}
TASK_IDS=${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}
NUM_TRIALS=${NUM_TRIALS:-50}
CPU_FRACTION=${CPU_FRACTION:-0.32}   # 4 delays x b_t=10 -> 40 envs on a 128-core host

cd "$REL"
for JOB in "${SUITES[@]}"; do
  SUITE=${JOB%%:*}; MAXSTEPS=${JOB##*:}
  OUT=$OUTBASE/${SUITE}_d${DELAYS}
  # concurrent runs on different GPUs need distinct cache dirs or inductor/triton corrupt each other
  CACHE=$OUTBASE/_cache/${SUITE}
  mkdir -p "$OUT" "$CACHE/tmp" "$CACHE/ind" "$CACHE/tri"
  echo "[eval_pi05] $SUITE (max_steps=$MAXSTEPS) -> $OUT"
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
  TMPDIR=$CACHE/tmp TORCHINDUCTOR_CACHE_DIR=$CACHE/ind TRITON_CACHE_DIR=$CACHE/tri \
  "$PY" eval/driver.py \
    --backbone pi05 --policy-path "$POLICY" \
    --predictor-checkpoint "$CKPT" --corrector-checkpoint "$CORRECTOR" \
    ${TOKENIZER:+--tokenizer-path "$TOKENIZER"} \
    --delays "$DELAYS" --task-suite-name "$SUITE" --task-ids "$TASK_IDS" \
    --num-trials-per-task "$NUM_TRIALS" --max-steps "$MAXSTEPS" \
    --preinfer-steps 25 --cpu-fraction "$CPU_FRACTION" \
    --output-dir "$OUT" --device cuda --resume
done
echo "[eval_pi05] all suites done -> $OUTBASE"
