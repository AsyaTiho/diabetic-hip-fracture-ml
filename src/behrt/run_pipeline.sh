#!/bin/bash
PYTHON="${PYTHON:-python}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(date)] Waiting for train_behrt_UPDATED.py to finish..."
while pgrep -f train_behrt_UPDATED.py > /dev/null; do sleep 60; done
echo "[$(date)] Training done. Waiting 60s for memory to free..."
sleep 60

echo "[$(date)] Starting build_finetune_dataset.py..."
cd $DIR && $PYTHON build_finetune_dataset.py > build_finetune.log 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] build_finetune_dataset.py failed. Aborting."
    exit 1
fi
echo "[$(date)] Build done. Waiting 60s for memory to free..."
sleep 60

echo "[$(date)] Starting fine-tuning..."
cd $DIR && $PYTHON finetune_behrt_lora.py > finetune.log 2>&1
echo "[$(date)] Fine-tuning done."
