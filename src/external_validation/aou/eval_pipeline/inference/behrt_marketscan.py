"""Re-run BEHRT-LoRA inference on the MarketScan TEST split.

Reuses FinetuneDataset + BehrtForSequenceClassification + all hyperparams from
finetune_behrt_lora.py so the test split is byte-identical (SEED=35, TEST_FRAC=0.10,
stratified on risk_class).

Output: outputs/predictions/marketscan_behrt.npz
  keys: y_true (int64), y_pred (int64), y_proba (N, 4 float32), enrolid (int64)
"""

import os
import gc
import sys
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# Import dataset + model + constants from the training script (still in embeddings_BEHRT)
sys.path.insert(0, os.environ.get('HIPFX_BEHRT_CODE', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'behrt')))
from finetune_behrt_lora import (
    FinetuneDataset, BehrtForSequenceClassification,
    DATA_DIR, OUTPUT_DIR, FINETUNE_DATA, PRETRAIN_CKPT, FINETUNE_CKPT,
    MAX_LEN, NUM_LABELS, BATCH_SIZE,
    SEED, TEST_FRAC, VAL_FRAC,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)
from MLM_model import BehrtConfig, BehrtForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETSCAN_PRED_OUT

OUT_PATH = MARKETSCAN_PRED_OUT['BEHRT-LoRA']


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 1
    print(f'Device: {device}  (GPUs: {num_gpus})\n')

    print('Loading vocabularies...')
    with open(os.path.join(DATA_DIR, 'token2idx.pkl'), 'rb') as f:
        token2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'age2idx.pkl'), 'rb') as f:
        age2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'patient_info_db1.pkl'), 'rb') as f:
        patient_info = pickle.load(f)
    sex_dict = {eid: info['sex'] for eid, info in patient_info.items()}
    del patient_info
    gc.collect()

    print('Loading fine-tune dataset...')
    sequences = pd.read_pickle(FINETUNE_DATA)
    print(f'  Patients: {len(sequences):,}')

    trainval_df, test_df = train_test_split(
        sequences, test_size=TEST_FRAC, random_state=SEED,
        stratify=sequences['risk_class'],
    )
    test_df = test_df.reset_index(drop=True)
    del sequences, trainval_df
    gc.collect()
    print(f'  TEST patients: {len(test_df):,}')

    saved = np.load(os.path.join(DATA_DIR, 'test_enrolids_2026-06-08.npy'))
    assert np.array_equal(np.sort(saved), np.sort(test_df['enrolid'].values.astype(np.int64))), \
        'TEST split does not match saved test_enrolids — SEED/split logic drifted!'
    print('  Split matches saved test_enrolids.npy')

    enrolids = test_df['enrolid'].values.astype(np.int64)
    test_ds  = FinetuneDataset(test_df, token2idx, age2idx, MAX_LEN, sex_dict)

    eff_batch = BATCH_SIZE * num_gpus * 2
    loader = DataLoader(test_ds, batch_size=eff_batch, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    print(f'\nLoading pre-trained backbone from {PRETRAIN_CKPT}...')
    ckpt = torch.load(PRETRAIN_CKPT, map_location='cpu', weights_only=False)
    config = BehrtConfig(**ckpt['config_dict'])
    mlm_model = BehrtForMaskedLM(config)
    mlm_model.load_state_dict(ckpt['model_state_dict'])
    bert_backbone = mlm_model.bert

    print('Applying LoRA adapters...')
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias='none',
    )
    get_peft_model(bert_backbone, lora_config)

    model = BehrtForSequenceClassification(
        bert=bert_backbone,
        hidden_size=config.hidden_size,
        num_labels=NUM_LABELS,
        dropout=config.hidden_dropout_prob,
    )

    print(f'Loading LoRA fine-tune checkpoint from {FINETUNE_CKPT}...')
    lora_ckpt = torch.load(FINETUNE_CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt['model_state_dict'])
    print(f'  Fine-tune ckpt: epoch {lora_ckpt.get("epoch", "?")}, '
          f'val_loss={lora_ckpt.get("val_loss", float("nan")):.4f}')

    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.eval()

    print(f'\nRunning inference on TEST set ({len(test_ds):,} patients)...')
    use_amp = (device.type == 'cuda')
    all_logits, all_labels = [], []

    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Test'):
            age, code, pos, seg, mask, sex, labels = [t.to(device) for t in batch]
            with autocast(enabled=use_amp):
                logits = model(code, age, seg, pos, sex, mask)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy().astype(np.int64)
    y_proba    = torch.softmax(all_logits, dim=1).numpy().astype(np.float32)
    y_pred     = all_logits.argmax(dim=1).numpy().astype(np.int64)
    print(f'Inference done in {time.time() - t0:.1f}s')

    assert len(all_labels) == len(enrolids), \
        f'Length mismatch: logits {len(all_labels)} vs enrolids {len(enrolids)}'

    print('\nPer-class test counts:')
    for c in range(NUM_LABELS):
        n_true = int((all_labels == c).sum())
        n_pred = int((y_pred == c).sum())
        print(f'  class {c}: true={n_true:>8,}  pred={n_pred:>8,}')

    np.savez(OUT_PATH,
             y_true=all_labels, y_pred=y_pred, y_proba=y_proba,
             enrolid=enrolids)
    print(f'\nSaved -> {OUT_PATH}')


if __name__ == '__main__':
    main()
