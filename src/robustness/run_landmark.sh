#!/usr/bin/env bash
# BEHRT_AGE_MODE=normal: this arm KEEPS age. The fixed observation window
# already removes the age-gap signal (an age-only predictor scores 0.505 here
# against 0.903 on the cut-at-fracture design), and age is a genuine
# hip-fracture risk factor. This matches the arm reported in the manuscript.
# Fine-tune the two landmark arms (delta = 30 and delta = 180) for the Discussion
# "Robustness of the risk signal" analysis. Reuses the unified LoRA fine-tune
# (../behrt/finetune_behrt_lora.py) via environment variables: the MLM backbone is
# reused and only the LoRA adapters + classification head are re-fit. The age
# channel is selected per arm via BEHRT_AGE_MODE, set below.
#
# Fresh per-arm output dirs, so nothing published is touched. Arm delta=30 then
# arm delta=180, sequentially. A CUDA GPU is required.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
DATA_DIR="${HIPFX_BEHRT_DIR:-data/behrt}"
FINETUNE="../behrt/finetune_behrt_lora.py"

run_arm () {   # $1 = delta
  D=$1
  echo; echo "########## ARM landmark delta=$D  $(date '+%F %T') ##########"
  BEHRT_FINETUNE_DATA="$DATA_DIR/patient_sequences_landmark_${D}.pkl" \
  BEHRT_SPLIT_FILE="$DATA_DIR/split_landmark_${D}.npz" \
  BEHRT_OUTPUT_DIR="$DATA_DIR/output_landmark_${D}" \
  BEHRT_AGE_MODE=normal \
  "$PYTHON" -u "$FINETUNE"
  echo "########## ARM delta=$D exit code $?  $(date '+%F %T') ##########"
}

run_arm 30
run_arm 180
echo "ALL LANDMARK FINE-TUNES DONE $(date '+%F %T')"
