"""Re-run BEHRT-LoRA inference on the UChicago external validation cohort.

The 2026-05-22 npz at uchicago_output/uchicago_frax_behrt_results_2026-05-22.npz
is from the pre-BI_181-cleanup model. This script produces a fresh prediction
file from the post-rebuild LoRA-tuned model (output/behrt_lora_best.pt).

Output: outputs/predictions/uchicago_behrt.npz
  Full 4-class schema (consumed by tables/figures):
    y_true (int64)   — risk_class label from uchicago_dm_patient_sequences.pkl
    y_pred (int64)   — argmax of softmax
    y_proba (N, 4 float32)
    enrolid (int64)  — UChicago studyid
  + clinical-comparison arrays (used by figUC_A/B/C):
    uc_risk_score (float64) — 1 - P(class 0), restricted to FRAX-having subset
    frax_hip (float64)
    y_hip (int64)           — binary hip-fracture outcome
"""

import os
import gc
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

# Reuse model classes + hyperparams from the finetune script
sys.path.insert(0, os.environ.get('HIPFX_BEHRT_CODE', str(Path(__file__).resolve().parents[4] / 'behrt')))
from finetune_behrt_lora import (
    FinetuneDataset, BehrtForSequenceClassification,
    DATA_DIR, PRETRAIN_CKPT, FINETUNE_CKPT,
    MAX_LEN, NUM_LABELS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)
from MLM_model import BehrtConfig, BehrtForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UCHICAGO_PRED_OUT, CLASS_NAMES_NATIVE

UCHICAGO_DIR    = Path(os.environ.get('HIPFX_UCHICAGO_DIR', 'data/uchicago'))
UCHICAGO_OUTPUT = Path(DATA_DIR) / 'uchicago_output'
BATCH_SIZE      = 256  # UChicago is ~13.5k patients; small batch is fine


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 1
    print(f'Device: {device}  (GPUs: {num_gpus})\n')

    print('Loading vocabularies...')
    with open(os.path.join(DATA_DIR, 'token2idx.pkl'), 'rb') as f:
        token2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'age2idx.pkl'), 'rb') as f:
        age2idx = pickle.load(f)

    print('\nLoading UChicago data...')
    uc_sequences = pd.read_pickle(UCHICAGO_OUTPUT / 'uchicago_dm_patient_sequences.pkl')
    print(f'  UChicago sequences: {len(uc_sequences):,} patients')
    with open(UCHICAGO_OUTPUT / 'uchicago_dm_sex_dict.pkl', 'rb') as f:
        uc_sex_dict = pickle.load(f)

    print('\nUChicago risk_class distribution:')
    for cls in range(NUM_LABELS):
        n = int((uc_sequences['risk_class'] == cls).sum())
        print(f'  class {cls} ({CLASS_NAMES_NATIVE[cls]:<22s}): {n:>6,}')

    uc_outcome = pd.read_csv(UCHICAGO_DIR / 'Anony DM Outcome and FRAX.csv')
    uc_outcome = uc_outcome.rename(columns={'studyid': 'enrolid'})
    uc_eval = uc_sequences.merge(uc_outcome, on='enrolid', how='inner')
    print(f'\nUChicago eval set (sequences ∩ outcomes): {len(uc_eval):,} patients')

    # ── Rebuild model exactly like finetune did ─────────────────
    print(f'\nLoading backbone from {PRETRAIN_CKPT}...')
    ckpt = torch.load(PRETRAIN_CKPT, map_location='cpu', weights_only=False)
    config = BehrtConfig(**ckpt['config_dict'])
    mlm_model = BehrtForMaskedLM(config)
    mlm_model.load_state_dict(ckpt['model_state_dict'])
    bert_backbone = mlm_model.bert
    del ckpt, mlm_model; gc.collect()

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
    del lora_ckpt; gc.collect()

    model = model.to(device)
    model.eval()

    # ── Build dataloader and run inference ──────────────────────
    print('\nBuilding DataLoader...')
    uc_dataset = FinetuneDataset(uc_eval, token2idx, age2idx, MAX_LEN,
                                 sex_dict=uc_sex_dict)
    loader = DataLoader(uc_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))
    print(f'  {len(loader)} batches of {BATCH_SIZE}')

    print('\nRunning inference...')
    all_logits, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            age, code, pos, seg, mask, sex, labels = [t.to(device) for t in batch]
            logits = model(code, age, seg, pos, sex, mask)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    y_true   = torch.cat(all_labels).numpy().astype(np.int64)
    y_proba  = torch.softmax(all_logits, dim=1).numpy().astype(np.float32)
    y_pred   = all_logits.argmax(dim=1).numpy().astype(np.int64)
    enrolids = uc_eval['enrolid'].values.astype(np.int64)
    risk_all = 1.0 - y_proba[:, 0].astype(np.float64)  # full N
    print(f'Inference done in {time.time() - t0:.1f}s  ({len(y_true):,} patients)')

    # ── Subset patients with FRAX scores (for figUC_*) ──────────
    frax_mask = uc_eval['frax_hip_adj'].notna().values
    n_frax = int(frax_mask.sum())
    uc_risk_score = risk_all[frax_mask]
    frax_hip = uc_eval.loc[frax_mask, 'frax_hip_adj'].astype(np.float64).values
    y_hip    = uc_eval.loc[frax_mask, 'nt_fx_hip'].astype(np.int64).values
    print(f'\nFRAX-having subset (for clinical figures): N = {n_frax:,}')
    print(f'  hip fractures = {int(y_hip.sum())} ({100*y_hip.mean():.2f}%)')
    print(f'  BEHRT risk range: [{uc_risk_score.min():.4f}, {uc_risk_score.max():.4f}]')
    print(f'  FRAX hip % range: [{frax_hip.min():.2f}, {frax_hip.max():.2f}]')

    np.savez(UCHICAGO_PRED_OUT,
             y_true=y_true, y_pred=y_pred, y_proba=y_proba, enrolid=enrolids,
             uc_risk_score=uc_risk_score, frax_hip=frax_hip, y_hip=y_hip)
    print(f'\nSaved -> {UCHICAGO_PRED_OUT}')


if __name__ == '__main__':
    main()
