#!/usr/bin/env bash
# Fine-tune the two landmark arms (delta = 30 and delta = 180) for the Discussion
# "Robustness of the risk signal" analysis. Reuses the unified LoRA fine-tune
# (../behrt/finetune_behrt_lora.py) via environment variables: the MLM backbone is
# reused and only the LoRA adapters + classification head are re-fit. Age is always
# blanked by that script, so there is nothing to configure for age here.
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
  "$PYTHON" -u "$FINETUNE"
  echo "########## ARM delta=$D exit code $?  $(date '+%F %T') ##########"
}

run_arm 30
run_arm 180
echo "ALL LANDMARK FINE-TUNES DONE $(date '+%F %T')"
