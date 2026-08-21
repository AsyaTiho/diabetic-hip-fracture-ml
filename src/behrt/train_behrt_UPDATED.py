#!/usr/bin/env python
"""BEHRT MLM pre-training with Optuna hyperparameter optimization.

Phase 1 — Bayesian hyperparameter search (Optuna) on a data subset. -> will skip this step and use the best hyperparameters found in the original script.
Phase 2 — Full training with the best hyperparameters.
Phase 3 — Extract patient embeddings from the trained model.

Usage:
    python train_behrt.py                        # run interactively
    nohup python train_behrt.py > train.log &    # run in background

Improvements over original:
    - Multi-GPU training via DataParallel (all 4x Quadro RTX 8000s)
    - Mixed precision training (torch.cuda.amp) for ~2x speedup
    - Gradient accumulation to simulate larger effective batch size
    - Mid-epoch evaluation (4x per epoch) for faster feedback
    - Early stopping based on mid-epoch checkpoints
    - Resume from checkpoint if training is interrupted
"""

import gc
import os
import random
import sys
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler          # NEW: mixed precision
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import optuna
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MLM_model import BehrtConfig, BehrtForMaskedLM
from MLM import MLMLoader
from optimiser_model import create_optimizer_and_scheduler

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these to suit your setup
# ═══════════════════════════════════════════════════════════════
DATA_DIR   = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

MAX_LEN    = 512
# BATCH_SIZE is per-GPU. With 4 GPUs the effective per-step batch before
# accumulation is BATCH_SIZE * NUM_GPUS = 64 * 4 = 256.
BATCH_SIZE  = 64
SEED        = 42

# Number of DataLoader worker processes per GPU.
# 4 GPUs * 2 workers = 8 total CPU workers prefetching data.
NUM_WORKERS = 2

# NEW: Gradient accumulation
# Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS = 64 * 4 = 256
# Increase ACCUMULATION_STEPS if you want an even larger effective batch.
# Decrease if training is slower than expected (less overhead).
ACCUMULATION_STEPS = 4

# NEW: Mid-epoch evaluation frequency
# 4 means we evaluate 4 times per epoch (every 25% of batches)
# Increase for more frequent feedback, decrease to reduce eval overhead
EVALS_PER_EPOCH = 4

# -- Full training --
FINAL_EPOCHS = 100

# If True, refuse to resume from an existing checkpoint and force a fresh
# training run.  When True the script ABORTS if a checkpoint is found at
# OUTPUT_DIR/behrt_best.pt -- move the old checkpoint into
# ~/wrong_or_old_runs/output/ first.  (Added 2026-06-07 after the BI_181
# data-contamination fix made all prior checkpoints unusable.)
START_FROM_SCRATCH = True

# NEW: PATIENCE now counts evaluations, not epochs.
# With EVALS_PER_EPOCH=4, PATIENCE=10 means stop after 2.5 epochs
# of no improvement. Adjust upward if you want more tolerance.
PATIENCE = 10

# -- Data split --
VAL_FRAC = 0.1


# ═══════════════════════════════════════════════════════════════
#  INFERENCE DATASET (no masking — for embedding extraction)
# ═══════════════════════════════════════════════════════════════
class InferenceDataset(Dataset):
    """Mirrors MLMLoader preprocessing but skips random masking.
    Stores data as int32 numpy arrays to avoid copy-on-write in workers.
    """

    def __init__(self, dataframe, token2idx, age2idx, max_len, sex_dict=None,
                 code='code', age='age'):
        self.max_len = max_len
        self.cls_id  = token2idx['CLS']
        self.sep_id  = token2idx['SEP']
        self.pad_id  = token2idx['PAD']
        self.age_pad_id = age2idx.get('PAD', 0)
        unk = token2idx.get('UNK', 0)

        self.code_ids = [
            np.array([token2idx.get(c, unk) for c in row], dtype=np.int32)
            for row in dataframe[code]
        ]
        self.age_ids = [
            np.array([age2idx.get(str(a), 0) for a in row], dtype=np.int32)
            for row in dataframe[age]
        ]

        # Store sex values (static per patient, will be broadcast to all positions)
        if sex_dict is not None and 'enrolid' in dataframe.columns:
            self.sex_values = np.array([
                sex_dict.get(enrolid, 0) for enrolid in dataframe['enrolid']
            ], dtype=np.int32)
        else:
            self.sex_values = np.zeros(len(dataframe), dtype=np.int32)

    def __getitem__(self, index):
        ml = self.max_len
        code = self.code_ids[index][-(ml - 1):]
        age  = self.age_ids[index][-(ml - 1):]

        if code[0] != self.sep_id:
            code = np.concatenate(([self.cls_id], code))
            age  = np.concatenate(([age[0]], age))
        else:
            code = code.copy()
            code[0] = self.cls_id

        seq_len = len(code)

        mask = np.ones(ml, dtype=np.int32)
        mask[seq_len:] = 0

        age_padded = np.full(ml, self.age_pad_id, dtype=np.int32)
        age_padded[:seq_len] = age[:seq_len]

        # Broadcast sex to all positions (static feature)
        sex_broadcast = np.full(ml, self.sex_values[index], dtype=np.int32)

        token_ids = np.full(ml, self.pad_id, dtype=np.int32)
        token_ids[:seq_len] = code[:seq_len]

        position = np.zeros(ml, dtype=np.int32)
        segment  = np.zeros(ml, dtype=np.int32)
        pos_counter = 0
        seg_flag = 0
        for i in range(ml):
            tok = token_ids[i]
            position[i] = pos_counter
            segment[i]  = seg_flag
            if tok == self.sep_id:
                pos_counter += 1
                seg_flag = 1 - seg_flag

        return (torch.LongTensor(age_padded),
                torch.LongTensor(token_ids),
                torch.LongTensor(position),
                torch.LongTensor(segment),
                torch.LongTensor(mask),
                torch.LongTensor(sex_broadcast))

    def __len__(self):
        return len(self.code_ids)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_one_epoch(model, loader, optimizer, scheduler, device,
                    scaler, val_loader, best_val, wait,
                    eval_steps, desc='Train',
                    save_path=None, epoch=None, best_params=None, config=None,
                    mid_val_loader=None):
    """
    Train for one epoch with:
      - Mixed precision (autocast + GradScaler)
      - Gradient accumulation (ACCUMULATION_STEPS)
      - Mid-epoch evaluation every eval_steps batches (uses mid_val_loader if given)
      - Mid-epoch checkpoint saving when val loss improves (if save_path given)

    Returns updated (train_loss, best_val, wait, improved_this_epoch)
    """
    model.train()
    total_loss     = 0.0
    n              = 0
    improved       = False
    use_amp        = (device.type == 'cuda')  # True on your GPU; safe fallback for other devices

    pbar = tqdm(loader, desc=desc, leave=False)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar, start=1):
        age, code, pos, seg, mask, label, sex = [t.to(device) for t in batch]

        # ── forward pass (float16 on CUDA, float32 elsewhere) ──
        with autocast(enabled=use_amp):
            loss, _, _ = model(code, age_ids=age, seg_ids=seg,
                               posi_ids=pos, sex_ids=sex, attention_mask=mask,
                               masked_lm_labels=label)
            # Scale loss for accumulation: gradients are summed over
            # ACCUMULATION_STEPS batches, so we divide here so the
            # effective gradient equals the mean across all steps.
            loss = loss / ACCUMULATION_STEPS

        # ── backward pass (scaled to avoid float16 underflow) ──
        # DataParallel returns a 1D tensor of per-replica losses; take mean to get a scalar
        loss_scalar = loss.mean()
        scaler.scale(loss_scalar).backward()

        total_loss += loss_scalar.item() * ACCUMULATION_STEPS  # undo scaling for logging
        n += 1
        pbar.set_postfix(loss=f'{total_loss / n:.4f}')

        # ── optimizer step every ACCUMULATION_STEPS batches ──
        if step % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            # .module unwraps DataParallel to access the underlying model parameters
            nn.utils.clip_grad_norm_(
                model.module.parameters() if hasattr(model, 'module')
                else model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # ── mid-epoch evaluation (uses smaller subset if available) ──
        if step % eval_steps == 0:
            _eval_loader = mid_val_loader if mid_val_loader is not None else val_loader
            vl = evaluate(model, _eval_loader, device,
                          desc=f'  Val@step{step}')
            model.train()  # switch back to train mode after eval
            print(f'\n  [mid-epoch step {step}] val_loss={vl:.4f}  '
                  f'(best so far: {best_val:.4f})')

            if vl < best_val:
                best_val = vl
                wait     = 0
                improved = True
                if save_path is not None:
                    save_checkpoint(
                        save_path, model, optimizer, scheduler, scaler,
                        epoch, step, best_val, wait, best_params, config,
                    )
                    print(f'  -> saved mid-epoch checkpoint (step {step})')
            else:
                wait += 1

    # ── handle any leftover accumulation steps at end of epoch ──
    if n % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.module.parameters() if hasattr(model, 'module')
            else model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(n, 1), best_val, wait, improved


@torch.no_grad()
def evaluate(model, loader, device, desc='Val'):
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = (device.type == 'cuda')
    for batch in tqdm(loader, desc=desc, leave=False):
        age, code, pos, seg, mask, label, sex = [t.to(device) for t in batch]
        with autocast(enabled=use_amp):
            loss, _, _ = model(code, age_ids=age, seg_ids=seg,
                               posi_ids=pos, sex_ids=sex, attention_mask=mask,
                               masked_lm_labels=label)
        # DataParallel may return per-replica losses; take mean to get a scalar
        loss_scalar = loss.mean()
        total_loss += loss_scalar.item()
        n += 1
    return total_loss / max(n, 1)


def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    epoch, step, best_val, wait, best_params, config):
    """Save full training state so we can resume exactly from this point.
    Uses .module to unwrap DataParallel before saving — the checkpoint
    contains plain model weights that can be loaded with or without DataParallel.
    """
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict':     raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'config_dict':          config.to_dict(),
        'epoch':                epoch,
        'step':                 step,
        'val_loss':             best_val,
        'wait':                 wait,
        'best_params':          best_params,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load full training state. Returns (epoch, best_val, wait).
    Handles both new-format checkpoints (with optimizer/scheduler/scaler state)
    and old-format checkpoints (model weights only, saved by the original script).
    Also handles architecture mismatches (e.g., when adding new embedding layers).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, 'module') else model

    # Try to load state dict, handling missing/unexpected keys
    try:
        raw_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        architecture_mismatch = False
    except RuntimeError as e:
        # Architecture mismatch - load what we can
        print(f'\n  ⚠️  Architecture mismatch detected!')
        print(f'  Error: {str(e)[:200]}...')
        print(f'  Loading compatible parameters only. New parameters will be randomly initialized.\n')

        # Load non-strict (skip missing/unexpected keys)
        missing_keys, unexpected_keys = raw_model.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )

        if missing_keys:
            print(f'  Missing keys (newly added): {missing_keys[:5]}...')
        if unexpected_keys:
            print(f'  Unexpected keys (removed): {unexpected_keys[:5]}...')

        architecture_mismatch = True

    if 'optimizer_state_dict' in ckpt and not architecture_mismatch:
        # New format — restore full training state (only if architecture matches)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        wait = ckpt.get('wait', 0)
        print(f'  Resumed from epoch {ckpt["epoch"]}, '
              f'best_val={ckpt["val_loss"]:.4f}, '
              f'patience_counter={wait}')
        return ckpt['epoch'], ckpt['val_loss'], wait
    else:
        # Old format OR architecture mismatch — start fresh
        wait = 0
        if architecture_mismatch:
            print(f'  Starting fresh training due to architecture changes.')
            print(f'  Previous checkpoint was from epoch {ckpt.get("epoch", "unknown")}.\n')
            return 0, float('inf'), 0  # Start from scratch
        else:
            print(f'  Loaded old-format checkpoint from epoch {ckpt["epoch"]}.')
            print(f'  Optimizer/scheduler state not found — starting fresh.')
            print(f'  best_val={ckpt["val_loss"]:.4f}')
            return ckpt['epoch'], ckpt['val_loss'], wait


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}\n')

    # ── load data ─────────────────────────────────────────────
    print('Loading data...')
    sequences = pd.read_pickle(os.path.join(DATA_DIR, 'patient_sequences.pkl'))
    with open(os.path.join(DATA_DIR, 'token2idx.pkl'), 'rb') as f:
        token2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'age2idx.pkl'), 'rb') as f:
        age2idx = pickle.load(f)

    # Load patient sex information
    with open(os.path.join(DATA_DIR, 'patient_info_db1.pkl'), 'rb') as f:
        patient_info = pickle.load(f)
    # Create sex dictionary: enrolid -> sex value (1 or 2)
    sex_dict = {enrolid: info['sex'] for enrolid, info in patient_info.items()}
    del patient_info  # Free memory
    gc.collect()

    print(f'  Patients:       {len(sequences):,}')
    print(f'  Vocab size:     {len(token2idx):,}')
    print(f'  Age vocab size: {len(age2idx):,}')
    print(f'  Sex vocab size: 3 (0=PAD, 1=male, 2=female)')

    # ── train / val split ─────────────────────────────────────
    train_idx, val_idx = train_test_split(
        range(len(sequences)), test_size=VAL_FRAC, random_state=SEED
    )
    train_df = sequences.iloc[train_idx].reset_index(drop=True)
    val_df   = sequences.iloc[val_idx].reset_index(drop=True)
    print(f'  Train: {len(train_df):,}  |  Val: {len(val_df):,}')

    train_dataset = MLMLoader(train_df, token2idx, age2idx, MAX_LEN, sex_dict=sex_dict)
    val_dataset   = MLMLoader(val_df,   token2idx, age2idx, MAX_LEN, sex_dict=sex_dict)

    del train_df, val_df, sequences
    gc.collect()

    # ══════════════════════════════════════════════════════════
    #  Load Best Hyperparameters
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  Load Best Hyperparameters')
    print('=' * 60 + '\n')
    with open(os.path.join(OUTPUT_DIR, 'optuna_best_params.pkl'), 'rb') as f:
        best = pickle.load(f)
    print(f'Best hyperparameters: {best}')

    config = BehrtConfig(
        vocab_size=len(token2idx),
        hidden_size=best['hidden_size'],
        num_hidden_layers=best['num_hidden_layers'],
        num_attention_heads=best['num_attention_heads'],
        intermediate_size=best['intermediate_size'],
        hidden_dropout_prob=best['dropout'],
        attention_probs_dropout_prob=best['dropout'],
        max_position_embeddings=MAX_LEN,
        seg_vocab_size=2,
        age_vocab_size=len(age2idx),
        sex_vocab_size=3,  # 0=PAD, 1=male, 2=female
    )
    model = BehrtForMaskedLM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {param_count:,}')

    # ── wrap model across all available GPUs ──────────────────
    # DataParallel splits each batch across GPUs, runs forward passes
    # in parallel, then averages the gradients. With 4x RTX 8000s this
    # gives ~4x throughput. The model itself lives on cuda:0 (primary GPU).
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f'  Using {num_gpus} GPUs via DataParallel')
        model = nn.DataParallel(model)
    else:
        print(f'  Using 1 GPU')

    # ── data loaders ──────────────────────────────────────────
    # With DataParallel, PyTorch splits BATCH_SIZE across GPUs automatically,
    # so each GPU processes BATCH_SIZE // num_gpus samples per step.
    # We scale up BATCH_SIZE so each GPU still sees 64 samples.
    effective_batch = BATCH_SIZE * num_gpus
    t_loader = DataLoader(train_dataset, batch_size=effective_batch,
                          shuffle=True, num_workers=NUM_WORKERS * num_gpus,
                          pin_memory=False)
    v_loader = DataLoader(val_dataset, batch_size=effective_batch,
                          num_workers=0, pin_memory=False)

    # Smaller val subset for mid-epoch evaluations (20% of val set)
    mid_eval_size = len(val_dataset) // 5
    random.seed(SEED)
    mid_eval_idx = random.sample(range(len(val_dataset)), mid_eval_size)
    mid_eval_subset = Subset(val_dataset, mid_eval_idx)
    v_loader_mini = DataLoader(mid_eval_subset, batch_size=effective_batch,
                               num_workers=0, pin_memory=False)
    print(f'  Mid-epoch val subset: {mid_eval_size:,} samples '
          f'({len(v_loader_mini):,} batches)')

    steps = len(t_loader) * FINAL_EPOCHS
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        {'lr': best['lr'], 'warmup_proportion': 0.1, 'weight_decay': 0.01},
        steps,
    )

    # NEW: GradScaler for mixed precision — active on CUDA (your GPU)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── compute how often to evaluate within each epoch ───────
    # eval_steps: evaluate every (total_batches / EVALS_PER_EPOCH) steps
    eval_steps = max(1, len(t_loader) // EVALS_PER_EPOCH)
    print(f'\n  GPUs in use                 : {num_gpus}')
    print(f'  Batch size per GPU          : {BATCH_SIZE}')
    print(f'  Batch size across all GPUs  : {effective_batch}')
    print(f'  Gradient accumulation steps : {ACCUMULATION_STEPS}')
    print(f'  Effective batch size        : {effective_batch * ACCUMULATION_STEPS}')
    print(f'  Mid-epoch evals per epoch   : {EVALS_PER_EPOCH}')
    print(f'  Evaluate every N batches    : {eval_steps}')
    print(f'  Early stopping patience     : {PATIENCE} evaluations\n')

    # ══════════════════════════════════════════════════════════
    #  PHASE 2 — Full Training
    # ══════════════════════════════════════════════════════════
    print('=' * 60)
    print('  PHASE 2: Full Training with Best Hyperparameters')
    print('=' * 60 + '\n')

    best_val   = float('inf')
    wait       = 0
    start_epoch = 1

    # ── resume from checkpoint if one exists ──────────────────
    resume_path = os.path.join(OUTPUT_DIR, 'behrt_best.pt')
    if START_FROM_SCRATCH:
        if os.path.exists(resume_path):
            raise RuntimeError(
                f'START_FROM_SCRATCH=True but a checkpoint exists at {resume_path}. '
                f'Move it into ~/wrong_or_old_runs/output/ before launching a fresh run.'
            )
        print('START_FROM_SCRATCH=True -- training BEHRT from scratch.')
    elif os.path.exists(resume_path):
        print(f'Found existing checkpoint at {resume_path}. Resuming...')
        start_epoch, best_val, wait = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch += 1  # continue from next epoch

    for epoch in range(start_epoch, FINAL_EPOCHS + 1):
        t0 = time.time()

        tl, best_val, wait, improved = train_one_epoch(
            model, t_loader, optimizer, scheduler, device,
            scaler, v_loader, best_val, wait,
            eval_steps, desc=f'Epoch {epoch}',
            save_path=resume_path, epoch=epoch,
            best_params=best, config=config,
            mid_val_loader=v_loader_mini,
        )

        # end-of-epoch full val evaluation
        vl = evaluate(model, v_loader, device, desc=f'Epoch {epoch} val')
        dt = time.time() - t0

        print(f'Epoch {epoch:>3}/{FINAL_EPOCHS}  '
              f'train={tl:.4f}  val={vl:.4f}  ({dt:.0f}s)')

        # save checkpoint if end-of-epoch val is best overall
        if vl < best_val:
            best_val = vl
            wait     = 0
            save_checkpoint(
                resume_path,
                model, optimizer, scheduler, scaler,
                epoch, len(t_loader), best_val, wait, best, config,
            )
            print('  -> saved best model')
        elif improved:
            # mid-epoch improvement was captured; still save full state
            save_checkpoint(
                resume_path,
                model, optimizer, scheduler, scaler,
                epoch, len(t_loader), best_val, wait, best, config,
            )
            print('  -> saved best model (mid-epoch improvement)')
        else:
            wait += 1
            print(f'  No improvement. Patience: {wait}/{PATIENCE}')
            if wait >= PATIENCE:
                print(f'  Early stopping after {PATIENCE} evaluations '
                      f'without improvement.')
                break

        gc.collect()
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    #  PHASE 3 — Extract Patient Embeddings
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('  PHASE 3: Extracting Patient Embeddings')
    print('=' * 60 + '\n')

    # Free training data from memory before loading full sequences
    del train_dataset, val_dataset, mid_eval_subset
    del t_loader, v_loader, v_loader_mini
    gc.collect()
    torch.cuda.empty_cache()

    # Reload sequences from disk (was deleted earlier to save RAM during training)
    print('  Reloading patient_sequences.pkl for embedding extraction...')
    sequences = pd.read_pickle(os.path.join(DATA_DIR, 'patient_sequences.pkl'))

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ckpt['model_state_dict'])
    raw_model.eval()

    inf_dataset = InferenceDataset(sequences, token2idx, age2idx, MAX_LEN, sex_dict=sex_dict)
    inf_loader  = DataLoader(inf_dataset, batch_size=BATCH_SIZE * num_gpus * 2,
                             num_workers=NUM_WORKERS * num_gpus, pin_memory=False)

    use_amp = (device.type == 'cuda')
    emb_list = []
    with torch.no_grad():
        for batch in tqdm(inf_loader, desc='Extracting embeddings'):
            age, code, pos, seg, mask, sex = [t.to(device) for t in batch]
            with autocast(enabled=use_amp):
                _, pooled = raw_model.bert(code, age_ids=age, seg_ids=seg,
                                           posi_ids=pos, sex_ids=sex, attention_mask=mask)
            emb_list.append(pooled.cpu().float().numpy())

    embeddings = np.concatenate(emb_list, axis=0)
    print(f'Embeddings shape: {embeddings.shape}')

    np.save(os.path.join(OUTPUT_DIR, 'patient_embeddings.npy'), embeddings)

    emb_df = pd.DataFrame(embeddings, index=sequences['enrolid'].values)
    emb_df.to_pickle(os.path.join(OUTPUT_DIR, 'patient_embeddings.pkl'))

    print(f'\nDone!')
    print(f'  Best hyperparameters:  {best}')
    print(f'  Best val loss:         {best_val:.4f}')
    print(f'  Embedding dimensions:  {embeddings.shape}')
    print(f'  Output directory:      {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()