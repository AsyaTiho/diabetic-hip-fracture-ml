#!/usr/bin/env bash
# BEHRT_AGE_MODE=normal: this arm KEEPS age. The fixed observation window
# already removes the age-gap signal (an age-only predictor scores 0.505 here
# against 0.903 on the cut-at-fracture design), and age is a genuine
# hip-fracture risk factor. This matches the arm reported in the manuscript.
# Fine-tune the MATCHED cohort (landmark delta=180, 1:k demographic matching) for
# the Discussion "Robustness of the risk signal" analysis. Same recipe as
# run_landmark.sh: reuses the unified LoRA fine-tune (../behrt/finetune_behrt_lora.py)
# via environment variables, backbone reused, only LoRA adapters + head re-fit, age
# always blanked by that script. Class weights auto-adjust to the matched
# distribution inside the fine-tune script (compute_class_weight 'balanced').
#
# Fresh output dir, so nothing published or from the other arms is touched.
# A CUDA GPU is required.
#
#   ./run_matched.sh        # k = 5 (default)
#   ./run_matched.sh 5
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
DATA_DIR="${HIPFX_BEHRT_DIR:-data/behrt}"
FINETUNE="../behrt/finetune_behrt_lora.py"

K=${1:-5}
echo; echo "########## MATCHED FINE-TUNE  k=$K  landmark delta=180  $(date '+%F %T') ##########"
BEHRT_FINETUNE_DATA="$DATA_DIR/matched_sequences_180_k${K}.pkl" \
BEHRT_SPLIT_FILE="$DATA_DIR/split_matched_180_k${K}.npz" \
BEHRT_OUTPUT_DIR="$DATA_DIR/output_matched_k${K}" \
BEHRT_AGE_MODE=normal \
"$PYTHON" -u "$FINETUNE"
echo "########## MATCHED k=$K exit code $?  $(date '+%F %T') ##########"
