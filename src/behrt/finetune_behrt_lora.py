#!/usr/bin/env python
"""Fine-tune BEHRT for risk class prediction using LoRA (Low-Rank Adaptation).

Loads the pre-trained BEHRT checkpoint produced by train_behrt_UPDATED.py,
injects LoRA adapters into the transformer attention layers (Q and V
projections), and trains a classification head to predict risk classes
(0 = low, 1 = medium, 2 = high).

All original BERT weights are frozen; only LoRA parameters and the
classification head are updated — typically ~1–2% of total parameters.

Usage:
    python finetune_behrt_lora.py
"""

import gc
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MLM_model import BehrtConfig, BehrtForMaskedLM

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATA_DIR      = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
# OUTPUT_DIR / FINETUNE_DATA / FINETUNE_CKPT are env-overridable so each cohort
# config (see configs/) can read its own sequences and write its own checkpoint
# without colliding with the others.
OUTPUT_DIR    = os.environ.get('BEHRT_OUTPUT_DIR', os.path.join(DATA_DIR, 'output'))
FINETUNE_DATA = os.environ.get('BEHRT_FINETUNE_DATA',
                               os.path.join(DATA_DIR, 'patient_sequences_finetune.pkl'))
# PRETRAIN_CKPT stays pinned to the base output dir: the MLM backbone is shared by
# every fine-tune, so age-removed and landmark runs must still find it there.
PRETRAIN_CKPT = os.path.join(DATA_DIR, 'output', 'behrt_best.pt')
FINETUNE_CKPT = os.environ.get('BEHRT_FINETUNE_CKPT',
                               os.path.join(OUTPUT_DIR, 'behrt_lora_best.pt'))

# Temporal-leakage correction, SELECTED PER ARM by BEHRT_AGE_MODE (required,
# no default; see resolve_age_mode below). A patient's age is deterministically
# related to the diabetes-to-fracture interval that defines the label, so an age
# embedding leaks the outcome. Every age token is therefore blanked before
# training and inference (see FinetuneDataset); this is the model reported in the
# manuscript. The leak itself is demonstrated separately in src/robustness/ (the
# age-gap cheater), not by shipping a runnable age-using model.

# Fixed train/val/test split for the landmark and matched sensitivity analyses.
# Those cohorts are strict-filtered, so re-deriving the split with
# train_test_split(random_state=SEED) would produce a different test set from the
# primary one and the arms would not be comparable patient-for-patient.
# BEHRT_SPLIT_FILE points at an .npz holding train/val/test enrolid arrays, built
# as the primary split minus the filtered patients. Unset => derive as before.
SPLIT_FILE    = os.environ.get('BEHRT_SPLIT_FILE')  # None => original behaviour

MAX_LEN            = 512
BATCH_SIZE         = 32      # per GPU
NUM_EPOCHS         = 50
LR                 = 2e-4    # LoRA typically uses a higher LR than full fine-tuning
WARMUP_PROPORTION  = 0.1
WEIGHT_DECAY       = 0.01
ACCUMULATION_STEPS = 4
PATIENCE           = 10      # epochs without val loss improvement before stopping
VAL_FRAC           = 0.10    # held-out for early stopping (was 0.15 pre-2026-06-08)
TEST_FRAC          = 0.10    # truly held-out for final eval; never seen during training
                             # (added 2026-06-08; matches XGB/NN train_*_4class.py exactly)
SEED               = 35      # matches XGB/NN SEED so test enrolids are identical
                             # across all 3 model pipelines (was 42 pre-2026-06-08)
NUM_WORKERS        = 2
NUM_LABELS         = 4       # risk classes (canonical encoding from
                             # _build_canonical_labels.py): 0=Control,
                             # 1=1-5y, 2=<1y, 3=5-10y to hip fracture
                             # after diabetes onset.

# If True, refuse to resume from an existing fine-tune checkpoint and force
# a fresh LoRA run.  When True the script ABORTS if a checkpoint exists at
# FINETUNE_CKPT -- move the old checkpoint into ~/wrong_or_old_runs/output/
# first.  (Added 2026-06-07.)  This only governs the LoRA fine-tune
# checkpoint; the pre-trained BEHRT base (PRETRAIN_CKPT) is still loaded
# normally -- LoRA always starts from the MLM-pretrained weights.
START_FROM_SCRATCH = True

# LoRA hyperparameters
LORA_R              = 16     # adapter rank
LORA_ALPHA          = 32     # scaling: effective scale = lora_alpha / r = 2
LORA_DROPOUT        = 0.1
# Inject LoRA into the Q and V projection matrices of every attention layer
LORA_TARGET_MODULES = ['query', 'value']


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════
def resolve_age_mode():
    """Return 'removed' or 'normal'; refuse if the caller did not say.

    REQUIRED, no default.  The arms deliberately differ and this file serves all
    of them:

      * The MAIN MarketScan model is age-blind.  Case records are cut at the
        fracture, so the age gap within a record encodes the diabetes-to-fracture
        interval that defines the label; a predictor given only that gap scores
        0.903.  Age must be withheld.
      * The LANDMARK and MATCHED arms keep age.  Every record is cut at a fixed
        interval after the diabetes index, so the age gap is the same for
        everyone and carries nothing (0.505, a coin flip).  Age is a genuine
        hip-fracture risk factor, so withholding it there would weaken the model
        for no benefit.

    This file previously blanked age unconditionally, with run_landmark.sh
    stating "there is nothing to configure for age here".  That reproduced an
    age-blind landmark arm, which is NOT the arm the manuscript reports.

    Resolved lazily, not at import, because several modules import this file for
    its model and dataset classes and must not be forced to set the variable.
    """
    m = os.environ.get('BEHRT_AGE_MODE')
    if m not in ('removed', 'normal'):
        raise SystemExit(
            'BEHRT_AGE_MODE must be set explicitly.\n'
            '  removed = every age token blanked to PAD. The main MarketScan\n'
            '            model, where records are cut at the fracture.\n'
            '  normal  = real ages. The landmark and matched arms, where the\n'
            '            fixed observation window already removes the signal.\n'
            f'  got: {m!r}')
    return m


class FinetuneDataset(Dataset):
    """Sequence → (tokens, ages, positions, segments, mask, sex, label).

    Preprocessing mirrors InferenceDataset in train_behrt_UPDATED.py but
    also returns the integer risk_class label for each patient.
    """

    def __init__(self, dataframe, token2idx, age2idx, max_len, sex_dict=None):
        self.max_len    = max_len
        self.cls_id     = token2idx['CLS']
        self.sep_id     = token2idx['SEP']
        self.pad_id     = token2idx['PAD']
        self.age_pad_id = age2idx.get('PAD', 0)
        unk             = token2idx.get('UNK', 0)

        self.code_ids = [
            np.array([token2idx.get(c, unk) for c in row], dtype=np.int32)
            for row in dataframe['code']
        ]
        # Age channel, per BEHRT_AGE_MODE (required; see resolve_age_mode).
        # 'removed' blanks every age token to the PAD index so the age embedding
        # contributes a constant, zero-information vector.  'normal' keeps the
        # real ages, which is what the landmark and matched arms use.
        if resolve_age_mode() == 'removed':
            self.age_ids = [
                np.full(len(row), self.age_pad_id, dtype=np.int32)
                for row in dataframe['age']
            ]
        else:
            self.age_ids = [
                np.array([age2idx.get(str(a), 0) for a in row], dtype=np.int32)
                for row in dataframe['age']
            ]

        if sex_dict is not None and 'enrolid' in dataframe.columns:
            self.sex_values = np.array(
                [sex_dict.get(eid, 0) for eid in dataframe['enrolid']],
                dtype=np.int32,
            )
        else:
            self.sex_values = np.zeros(len(dataframe), dtype=np.int32)

        self.labels = dataframe['risk_class'].values.astype(np.int64)

    def __len__(self):
        return len(self.code_ids)

    def __getitem__(self, index):
        ml   = self.max_len
        code = self.code_ids[index][-(ml - 1):]
        age  = self.age_ids[index][-(ml - 1):]

        # Prepend CLS token (copy-on-write safe)
        if code[0] != self.sep_id:
            code = np.concatenate(([self.cls_id], code))
            age  = np.concatenate(([age[0]], age))
        else:
            code    = code.copy()
            code[0] = self.cls_id

        seq_len = len(code)

        mask = np.ones(ml, dtype=np.int32)
        mask[seq_len:] = 0

        age_padded           = np.full(ml, self.age_pad_id, dtype=np.int32)
        age_padded[:seq_len] = age[:seq_len]

        sex_broadcast = np.full(ml, self.sex_values[index], dtype=np.int32)

        token_ids           = np.full(ml, self.pad_id, dtype=np.int32)
        token_ids[:seq_len] = code[:seq_len]

        # Build position and segment IDs (same logic as pre-training)
        position    = np.zeros(ml, dtype=np.int32)
        segment     = np.zeros(ml, dtype=np.int32)
        pos_counter = 0
        seg_flag    = 0
        for i in range(ml):
            position[i] = pos_counter
            segment[i]  = seg_flag
            if token_ids[i] == self.sep_id:
                pos_counter += 1
                seg_flag = 1 - seg_flag

        return (
            torch.LongTensor(age_padded),
            torch.LongTensor(token_ids),
            torch.LongTensor(position),
            torch.LongTensor(segment),
            torch.LongTensor(mask),
            torch.LongTensor(sex_broadcast),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════
class BehrtForSequenceClassification(nn.Module):
    """BEHRT backbone (with LoRA) + dropout + linear classification head.

    The bert attribute is a BehrtModel whose attention Q/V layers have
    been replaced in-place by LoRA wrappers via get_peft_model().
    Only LoRA parameters and the classifier head have requires_grad=True.
    """

    def __init__(self, bert, hidden_size: int, num_labels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, age_ids, seg_ids, posi_ids, sex_ids,
                attention_mask):
        _, pooled = self.bert(
            input_ids,
            age_ids=age_ids,
            seg_ids=seg_ids,
            posi_ids=posi_ids,
            sex_ids=sex_ids,
            attention_mask=attention_mask,
        )
        return self.classifier(self.dropout(pooled))


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(model, loader, device, class_weights, desc='Val'):
    """Return (val_loss, accuracy, weighted_f1, macro_auc)."""
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()

    loss_fn  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    use_amp  = (device.type == 'cuda')
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
        age, code, pos, seg, mask, sex, labels = [t.to(device) for t in batch]
        with autocast(enabled=use_amp):
            logits = model(code, age, seg, pos, sex, mask)
            loss   = loss_fn(logits, labels)
        total_loss  += loss.item()
        n           += 1
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)        # (N, num_labels)
    all_labels = torch.cat(all_labels)        # (N,)
    probs      = torch.softmax(all_logits, dim=1).numpy()
    preds      = all_logits.argmax(dim=1).numpy()
    labels_np  = all_labels.numpy()

    acc  = accuracy_score(labels_np, preds)
    f1   = f1_score(labels_np, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(labels_np, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')  # only one class present in small batches

    return total_loss / max(n, 1), acc, f1, auc


def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    epoch, best_val, wait, config_dict):
    raw = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict':     raw.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'config_dict':          config_dict,
        'epoch':                epoch,
        'val_loss':             best_val,
        'wait':                 wait,
    }, path)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f'Device: {device}\n')

    # ── load vocabularies ──────────────────────────────────────
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

    # ── load fine-tune dataset ─────────────────────────────────
    print('Loading fine-tune dataset...')
    sequences = pd.read_pickle(FINETUNE_DATA)
    print(f'  Patients:  {len(sequences):,}')
    print(f'  Columns:   {list(sequences.columns)}')
    print(f'\nClass distribution:')
    vc = sequences['risk_class'].value_counts().sort_index()
    for cls, cnt in vc.items():
        print(f'  class {cls}: {cnt:,} ({100 * cnt / len(sequences):.1f}%)')

    # ── train / val / test split  (3-way; SEED=35 matches XGB/NN exactly) ─
    if SPLIT_FILE:
        # Landmark / matched arms: load a FIXED split (the primary split minus the
        # patients strict filtration removed), so the test set is a strict subset
        # of the primary one and every arm scores on the same patients. The split
        # is loaded, not derived, so the primary test-enrolid sidecar is left alone.
        print(f'\n  Using fixed split from {SPLIT_FILE}')
        sp = np.load(SPLIT_FILE)
        by_id = sequences.set_index('enrolid')
        train_df = by_id.loc[sp['train']].reset_index()
        val_df   = by_id.loc[sp['val']].reset_index()
        test_df  = by_id.loc[sp['test']].reset_index()
        del by_id
    else:
        # First peel off TEST (10%) — never seen by training, used only by eval.
        trainval_df, test_df = train_test_split(
            sequences, test_size=TEST_FRAC, random_state=SEED,
            stratify=sequences['risk_class'],
        )
        # Then split remaining 90% into TRAIN and VAL.  test_size here is
        # VAL_FRAC / (1 - TEST_FRAC) so the final TRAIN/VAL/TEST proportions
        # are 80 / 10 / 10 of the total cohort.
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=VAL_FRAC / (1 - TEST_FRAC),
            random_state=SEED,
            stratify=trainval_df['risk_class'],
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        # Save TEST enrolids to a sidecar file so the evaluation scripts can build
        # the FinetuneDataset for the exact same held-out patients. (XGB/NN's
        # *_4class_results.npz files store y_true+y_pred+y_proba+enrolid for their
        # own test split; this enables apples-to-apples comparison.)
        test_enrolid_path = os.path.join(DATA_DIR, 'test_enrolids_2026-06-08.npy')
        np.save(test_enrolid_path, test_df['enrolid'].values.astype(np.int64))
        print(f'  Saved TEST enrolids to {test_enrolid_path}')

    print(f'\n  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}')

    train_ds = FinetuneDataset(train_df, token2idx, age2idx, MAX_LEN, sex_dict)
    val_ds   = FinetuneDataset(val_df,   token2idx, age2idx, MAX_LEN, sex_dict)

    del sequences, trainval_df, train_df, val_df, test_df
    gc.collect()

    # ── class weights for imbalanced loss ─────────────────────
    train_labels = np.array([train_ds.labels[i] for i in range(len(train_ds))])
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_LABELS),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    print(f'\nClass weights: {class_weights_np.round(3)}')

    # ── data loaders ──────────────────────────────────────────
    num_gpus        = torch.cuda.device_count() if device.type == 'cuda' else 1
    effective_batch = BATCH_SIZE * num_gpus

    t_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True,
                          num_workers=NUM_WORKERS * max(num_gpus, 1),
                          pin_memory=(device.type == 'cuda'))
    v_loader = DataLoader(val_ds,   batch_size=effective_batch * 2, shuffle=False,
                          num_workers=0, pin_memory=False)

    # ── load pre-trained checkpoint ───────────────────────────
    print(f'\nLoading pre-trained checkpoint from {PRETRAIN_CKPT}...')
    ckpt = torch.load(PRETRAIN_CKPT, map_location='cpu', weights_only=False)

    # Reconstruct config from the checkpoint's stored config_dict
    config_dict = ckpt['config_dict']
    config = BehrtConfig(**config_dict)

    # Build full MLM model and load pre-trained weights
    mlm_model = BehrtForMaskedLM(config)
    mlm_model.load_state_dict(ckpt['model_state_dict'])
    print(f'  Loaded from epoch {ckpt.get("epoch", "?")}  '
          f'(val_loss={ckpt.get("val_loss", float("nan")):.4f})')

    # Extract just the BERT backbone (discard the MLM head)
    bert_backbone = mlm_model.bert

    # ── inject LoRA into attention Q/V layers ─────────────────
    print('\nApplying LoRA adapters...')
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias='none',
    )
    # get_peft_model replaces target Linear layers in-place with LoRA wrappers
    # and sets requires_grad=False on all original parameters
    peft_bert = get_peft_model(bert_backbone, lora_config)
    peft_bert.print_trainable_parameters()

    # ── build classification model ────────────────────────────
    # bert_backbone's query/value layers are now LoRA-wrapped in-place;
    # only LoRA params + classifier head will have requires_grad=True
    model = BehrtForSequenceClassification(
        bert=bert_backbone,
        hidden_size=config.hidden_size,
        num_labels=NUM_LABELS,
        dropout=config.hidden_dropout_prob,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f'\nTotal parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,} '
          f'({100 * trainable_params / total_params:.2f}%)')

    if num_gpus > 1:
        print(f'Using {num_gpus} GPUs via DataParallel')
        model = nn.DataParallel(model)

    # ── optimizer — only trainable parameters ─────────────────
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay  = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_groups = [
        {
            'params': [p for n, p in trainable if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
        },
        {
            'params': [p for n, p in trainable if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    from torch.optim import AdamW
    optimizer = AdamW(optimizer_groups, lr=LR)

    total_steps   = len(t_loader) * NUM_EPOCHS // ACCUMULATION_STEPS
    warmup_steps  = int(total_steps * WARMUP_PROPORTION)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    print(f'\n  GPUs in use           : {num_gpus}')
    print(f'  Batch size per GPU    : {BATCH_SIZE}')
    print(f'  Effective batch size  : {effective_batch * ACCUMULATION_STEPS}')
    print(f'  Total optimizer steps : {total_steps:,}')
    print(f'  Warmup steps          : {warmup_steps:,}')
    print(f'  Early stopping        : {PATIENCE} epochs\n')

    # ── resume from fine-tune checkpoint if it exists ─────────
    best_val   = float('inf')
    wait       = 0
    start_epoch = 1

    if START_FROM_SCRATCH:
        if os.path.exists(FINETUNE_CKPT):
            raise RuntimeError(
                f'START_FROM_SCRATCH=True but a fine-tune checkpoint exists at '
                f'{FINETUNE_CKPT}. Move it into ~/wrong_or_old_runs/output/ '
                f'before launching a fresh LoRA run.'
            )
        print('START_FROM_SCRATCH=True -- LoRA fine-tuning from scratch.')
    elif os.path.exists(FINETUNE_CKPT):
        print(f'Found existing fine-tune checkpoint. Resuming...')
        resume = torch.load(FINETUNE_CKPT, map_location=device, weights_only=False)
        raw    = model.module if hasattr(model, 'module') else model
        raw.load_state_dict(resume['model_state_dict'])
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        scaler.load_state_dict(resume['scaler_state_dict'])
        best_val    = resume['val_loss']
        wait        = resume['wait']
        start_epoch = resume['epoch'] + 1
        print(f'  Resumed from epoch {resume["epoch"]}, '
              f'best_val={best_val:.4f}, patience={wait}')

    # ══════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════════════
    print('=' * 60)
    print('  Fine-tuning with LoRA')
    print('=' * 60 + '\n')

    loss_fn  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    use_amp  = (device.type == 'cuda')

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(t_loader, desc=f'Epoch {epoch}', leave=False), 1):
            age, code, pos, seg, mask, sex, labels = [t.to(device) for t in batch]

            with autocast(enabled=use_amp):
                logits = model(code, age, seg, pos, sex, mask)
                if isinstance(logits, torch.Tensor) and logits.dim() == 2:
                    loss = loss_fn(logits, labels)
                else:
                    # DataParallel may return a tuple; take mean if needed
                    loss = loss_fn(logits, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss.mean()).backward()
            total_loss += loss.item() * ACCUMULATION_STEPS
            n_steps    += 1

            if step % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                raw_model = model.module if hasattr(model, 'module') else model
                nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        # Handle leftover accumulation steps at end of epoch
        if n_steps % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            raw_model = model.module if hasattr(model, 'module') else model
            nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(n_steps, 1)

        # Validation
        val_loss, acc, f1, auc = evaluate(
            model, v_loader, device, class_weights,
            desc=f'Epoch {epoch} val',
        )
        dt = time.time() - t0

        print(f'Epoch {epoch:>3}/{NUM_EPOCHS}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'acc={acc:.4f}  wf1={f1:.4f}  auc={auc:.4f}  '
              f'({dt:.0f}s)')

        # Checkpoint and early stopping on val loss
        raw_model = model.module if hasattr(model, 'module') else model
        if val_loss < best_val:
            best_val = val_loss
            wait     = 0
            save_checkpoint(
                FINETUNE_CKPT, model, optimizer, scheduler, scaler,
                epoch, best_val, wait, config_dict,
            )
            print(f'  -> saved best model (val_loss={best_val:.4f})')
        else:
            wait += 1
            print(f'  No improvement. Patience: {wait}/{PATIENCE}')
            if wait >= PATIENCE:
                print(f'  Early stopping after {PATIENCE} epochs without improvement.')
                break

        gc.collect()
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    #  FINAL EVALUATION on best checkpoint
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  Final evaluation on best checkpoint')
    print('=' * 60)

    best_ckpt = torch.load(FINETUNE_CKPT, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(best_ckpt['model_state_dict'])

    val_loss, acc, f1, auc = evaluate(
        model, v_loader, device, class_weights, desc='Final val'
    )
    print(f'\nBest model (epoch {best_ckpt["epoch"]}):')
    print(f'  Val loss  : {val_loss:.4f}')
    print(f'  Accuracy  : {acc:.4f}')
    print(f'  Weighted F1: {f1:.4f}')
    print(f'  Macro AUC : {auc:.4f}')
    print(f'\nCheckpoint saved to: {FINETUNE_CKPT}')


if __name__ == '__main__':
    main()
