#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/chris/ultralytics"
COMPARE_ROOT="$ROOT/compares/MSAMNet-master"
LOG_DIR="$ROOT/runs/xt_compare_logs"
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(CSAUNet DNANet DnTNet MSAMNet)
fi

mkdir -p "$LOG_DIR"
export MPLCONFIGDIR=/tmp/matplotlib

source /home/chris/miniconda3/etc/profile.d/conda.sh
conda activate yolo

cd "$ROOT"

for model in "${MODELS[@]}"; do
  log_file="$LOG_DIR/${model}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] start $model" | tee -a "$log_file"
  python tools/xt/tools_run_compare_msod_x.py \
    --mode train \
    --model "$model" \
    --dataset-root "$ROOT/dataset/xt_seq_msod_x" \
    --epochs 100 \
    --gpus 0 \
    --workers 4 \
    --train-batch-size 4 \
    --test-batch-size 4 \
    --t-frame 3 \
    --input-size 512 \
    --suffix .tif \
    2>&1 | tee -a "$log_file"
  echo "[$(date '+%F %T')] done $model" | tee -a "$log_file"
done
