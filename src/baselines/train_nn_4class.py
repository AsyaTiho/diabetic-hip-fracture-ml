#!/usr/bin/env python
"""Train a sparse Neural Network for 4-class hip fracture risk prediction with Optuna HPO.

Architecture mirrors SparseClassifier from train_balanced_NN_Feb23.py, extended to 4-class.
Sparse→dense uses the batch-level CSR collate approach from that script
(one optimised matrix slice per batch, far faster than per-row conversion).

Multi-GPU strategy:
  - HPO runs on a single GPU (overhead not worth it for 150K subset).
  - Final training wraps the model in nn.DataParallel across all available GPUs.
    Effective batch = best['batch_size'] * num_gpus so each GPU sees the same
    per-device batch size as was tuned during HPO.

Classes (canonical encoding, matches _build_canonical_labels.py and the
BEHRT-LoRA fine-tune labels -- note classes 1 and 2 are swapped vs. the
natural chronological order, on purpose, so the two pipelines share the
same integer-to-meaning map):
  0 – Control (no hip fracture or > 10 yr)   ~99.3 %
  1 – Hip fracture 1–5 years                 ~0.48 %
  2 – Hip fracture within 1 year             ~0.31 %
  3 – Hip fracture 5–10 years                ~0.19 %

Outputs (all in same directory as this script):
  nn_4class_weights.pth
  optuna_study_nn_4class.pkl
  nn_4class_results.npz

Usage:
  cd Testing_NN_XGBoost_on_4_risk_classes
  python train_nn_4class.py
"""

import gc
import os
import time
from datetime import timedelta

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ── paths ──────────────────────────────────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE   = os.path.join(DATA_DIR, 'one_hot_features_4groups.npz')
META_FILE   = os.path.join(DATA_DIR, 'one_hot_labels_metadata_4groups.npz')
MODEL_OUT   = os.path.join(DATA_DIR, 'nn_4class_weights.pth')
STUDY_OUT   = os.path.join(DATA_DIR, 'optuna_study_nn_4class.pkl')
RESULTS_OUT = os.path.join(DATA_DIR, 'nn_4class_results.npz')

# ── config ─────────────────────────────────────────────────────
SEED = 35

CLASS0_CAP = 600_000

VAL_FRAC  = 0.10
TEST_FRAC = 0.10

HPO_SUBSET_SIZE = 150_000
N_TRIALS        = 30
HPO_EPOCHS      = 15
HPO_PATIENCE    = 4

FINAL_EPOCHS   = 60
FINAL_PATIENCE = 10
NUM_WORKERS    = 4

GROUP_LABELS = [
    'Control (no hip frac or >10yr)',
    'Hip frac 1–5 years',         # class 1
    'Hip frac within 1 year',     # class 2 (canonical encoding swap)
    'Hip frac 5–10 years',        # class 3
]
NUM_CLASSES = 4


# ── model ──────────────────────────────────────────────────────
class SparseClassifier(nn.Module):
    """4-layer MLP matching train_balanced_NN_Feb23.py, extended to 4-class output."""

    def __init__(self, input_size: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── dataset + collate ──────────────────────────────────────────
class SparseIndexDataset(Dataset):
    """Stores only row indices; batch-level CSR slice happens in collate_fn.
    Adapted from train_balanced_NN_Feb23.py — avoids per-row sparse access.
    """

    def __init__(self, indices: np.ndarray, y: np.ndarray):
        self.indices = indices
        self.y       = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.y[self.indices[idx]]


def make_collate_fn(X: csr_matrix):
    """Single optimised CSR batch slice → dense tensor per batch.
    Adapted from train_balanced_NN_Feb23.py.
    """
    def collate_fn(batch):
        orig_indices, targets = zip(*batch)
        dense    = X[list(orig_indices)].toarray().astype(np.float32)
        features = torch.from_numpy(dense)
        targets  = torch.tensor(np.array(targets), dtype=torch.long)
        return features, targets
    return collate_fn


def make_loader(X: csr_matrix, y: np.ndarray, indices: np.ndarray,
                batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Build a single DataLoader for one split. Train and val always use separate
    calls so they can reference different sparse matrices (X_train vs X_val)."""
    ds = SparseIndexDataset(indices, y)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        collate_fn=make_collate_fn(X),
        persistent_workers=(num_workers > 0),
    )


# ── helpers ────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def num_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def print_dist(y, label=''):
    unique, counts = np.unique(y, return_counts=True)
    print(f'  {label}')
    for cls, cnt in zip(unique, counts):
        print(f'    class {cls} ({GROUP_LABELS[cls]}): {cnt:,}  ({100*cnt/len(y):.2f}%)')


def downsample_class0(X, y, cap, rng):
    idx0     = np.where(y == 0)[0]
    idx_rest = np.where(y != 0)[0]
    if len(idx0) > cap:
        idx0 = rng.choice(idx0, size=cap, replace=False)
    keep = np.concatenate([idx0, idx_rest])
    rng.shuffle(keep)
    return X[keep], y[keep]


def class_weight_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    w = compute_class_weight(class_weight='balanced',
                             classes=np.arange(NUM_CLASSES), y=y)
    return torch.tensor(w, dtype=torch.float32).to(device)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        with autocast(enabled=(device.type == 'cuda')):
            logits = model(X_batch)
            loss   = loss_fn(logits, y_batch)
        total_loss += loss.item()
        n          += 1
        all_logits.append(logits.float().cpu())
        all_labels.append(y_batch.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = all_logits.argmax(dim=1).numpy()
    wf1   = f1_score(all_labels, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    return total_loss / max(n, 1), wf1, auc, preds, probs, all_labels


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            logits = model(X_batch)
            loss   = loss_fn(logits, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # clip on underlying parameters (works with DataParallel)
        raw = model.module if isinstance(model, nn.DataParallel) else model
        nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n          += 1
    return total_loss / max(n, 1)


# ── Optuna objective ───────────────────────────────────────────
def make_hpo_objective(X_hpo, y_hpo, device, input_size):

    def objective(trial):
        lr           = trial.suggest_float('lr',           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        dropout      = trial.suggest_float('dropout',      0.1,  0.5)
        batch_size   = trial.suggest_categorical('batch_size', [1024, 2048, 4096])

        idx = np.arange(len(y_hpo))
        tr_idx, va_idx = train_test_split(
            idx, test_size=0.2, stratify=y_hpo, random_state=SEED + trial.number,
        )

        cw      = class_weight_tensor(y_hpo[tr_idx], device)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        model   = SparseClassifier(input_size, NUM_CLASSES, dropout).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HPO_EPOCHS)
        scaler    = GradScaler(enabled=(device.type == 'cuda'))

        # HPO loaders share the same X_hpo matrix — both loaders reference it directly
        t_loader = make_loader(X_hpo, y_hpo, tr_idx, batch_size,
                               shuffle=True, num_workers=2)
        v_loader = make_loader(X_hpo, y_hpo, va_idx, batch_size * 2,
                               shuffle=False, num_workers=2)

        best_val = float('inf')
        wait     = 0
        for epoch in range(HPO_EPOCHS):
            train_one_epoch(model, t_loader, loss_fn, optimizer, scaler, device)
            scheduler.step()
            val_loss, *_ = evaluate(model, v_loader, loss_fn, device)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val:
                best_val = val_loss
                wait     = 0
            else:
                wait += 1
                if wait >= HPO_PATIENCE:
                    break

        del model, optimizer, scaler, t_loader, v_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return best_val

    return objective


# ── main ───────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng        = np.random.default_rng(SEED)
    device     = get_device()
    n_gpus     = num_gpus()
    load_start = time.time()

    print(f'Device: {device}  |  GPUs available: {n_gpus}')

    # 1. Load
    print('\nLoading data...')
    X    = load_npz(FEAT_FILE)
    meta = np.load(META_FILE, allow_pickle=True)
    y    = meta['y'].astype(np.int32)
    print(f'  X shape: {X.shape}  dtype: {X.dtype}')
    print_dist(y, 'Full dataset:')

    # 2. Stratified split — val/test keep full natural class distribution
    print('\nSplitting...')
    idx = np.arange(len(y))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y, test_size=TEST_FRAC, stratify=y, random_state=SEED,
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval, y_trainval,
        test_size=VAL_FRAC / (1 - TEST_FRAC),
        stratify=y_trainval, random_state=SEED,
    )
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    print_dist(y_val,  f'Val  (full dist, n={len(y_val):,}):')
    print_dist(y_test, f'Test (full dist, n={len(y_test):,}):')

    # 3. Downsample class 0 in training only
    print(f'\nDownsampling class 0 → {CLASS0_CAP:,}...')
    X_train, y_train = downsample_class0(X_train, y_train, CLASS0_CAP, rng)
    print_dist(y_train, f'Train (n={len(y_train):,}):')
    input_size = X_train.shape[1]

    load_time = time.time() - load_start
    print(f'Data ready in {timedelta(seconds=int(load_time))}')

    # 4. Build HPO subset
    print(f'\nBuilding HPO subset (n={HPO_SUBSET_SIZE:,})...')
    hpo_idx, _, y_hpo, _ = train_test_split(
        np.arange(len(y_train)), y_train,
        train_size=HPO_SUBSET_SIZE, stratify=y_train, random_state=SEED,
    )
    X_hpo = X_train[hpo_idx]
    print_dist(y_hpo, 'HPO subset:')

    # 5. Optuna HPO on single GPU
    print('\n' + '='*70)
    print(f'  Optuna HPO  ({N_TRIALS} trials, single GPU)')
    print('='*70 + '\n')

    study = optuna.create_study(
        direction  = 'minimize',
        study_name = 'nn_4class_optimization',
        sampler    = optuna.samplers.TPESampler(seed=SEED),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    optuna_start = time.time()
    study.optimize(
        make_hpo_objective(X_hpo, y_hpo, device, input_size),
        n_trials          = N_TRIALS,
        show_progress_bar = True,
    )
    optuna_time = time.time() - optuna_start

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f'\nHPO complete: {len(completed)} completed, {len(pruned)} pruned')
    print(f'Best val loss: {study.best_value:.6f}')
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v:.6g}' if isinstance(v, float) else f'  {k}: {v}')

    joblib.dump(study, STUDY_OUT)
    print(f'Study saved → {STUDY_OUT}')

    best = study.best_params
    del X_hpo
    gc.collect()

    # 6. Final model — all GPUs via DataParallel
    print('\n' + '='*70)
    print(f'  Training final model on {n_gpus} GPU(s) via DataParallel')
    print('='*70)

    # Scale effective batch by number of GPUs so per-device batch matches HPO
    per_device_batch = best['batch_size']
    effective_batch  = per_device_batch * n_gpus
    print(f'  Per-device batch: {per_device_batch}  ×  {n_gpus} GPUs  '
          f'= effective batch: {effective_batch}')

    cw      = class_weight_tensor(y_train, device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    print(f'  Class weights: {cw.cpu().numpy().round(3)}')

    raw_model   = SparseClassifier(input_size, NUM_CLASSES, best['dropout']).to(device)
    final_model = nn.DataParallel(raw_model) if n_gpus > 1 else raw_model

    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=best['lr'], weight_decay=best['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Train and val loaders reference DIFFERENT sparse matrices — built separately
    t_loader = make_loader(X_train, y_train, np.arange(len(y_train)),
                           batch_size=effective_batch, shuffle=True,
                           num_workers=NUM_WORKERS)
    v_loader = make_loader(X_val, y_val, np.arange(len(y_val)),
                           batch_size=effective_batch * 2, shuffle=False,
                           num_workers=NUM_WORKERS)

    t_start  = time.time()
    best_val = float('inf')
    wait     = 0

    for epoch in range(1, FINAL_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(final_model, t_loader, loss_fn, optimizer, scaler, device)
        val_loss, wf1, auc, *_ = evaluate(final_model, v_loader, loss_fn, device)
        scheduler.step(val_loss)

        print(f'Epoch {epoch:>3}/{FINAL_EPOCHS}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'wf1={wf1:.4f}  auc={auc:.4f}  ({time.time()-t0:.0f}s)')

        if val_loss < best_val:
            best_val = val_loss
            wait     = 0
            torch.save(raw_model.state_dict(), MODEL_OUT)
            print(f'  -> saved (val_loss={best_val:.4f})')
        else:
            wait += 1
            if wait >= FINAL_PATIENCE:
                print(f'  Early stopping after {FINAL_PATIENCE} epochs without improvement.')
                break
        gc.collect()

    final_train_time = time.time() - t_start

    # 7. Final evaluation on val + test using best checkpoint
    print('\n' + '='*70)
    print('  Final evaluation (best checkpoint)')
    print('='*70)

    raw_model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True))
    eval_model = nn.DataParallel(raw_model) if n_gpus > 1 else raw_model

    te_loader = make_loader(X_test, y_test, np.arange(len(y_test)),
                            batch_size=effective_batch * 2, shuffle=False,
                            num_workers=0)

    for split_name, loader in [('Val', v_loader), ('Test', te_loader)]:
        _, wf1, auc, preds, probs, labels_np = evaluate(eval_model, loader, loss_fn, device)
        mf1 = f1_score(labels_np, preds, average='macro', zero_division=0)
        print(f'\n── {split_name} ──')
        print(f'  Weighted F1: {wf1:.4f}')
        print(f'  Macro F1   : {mf1:.4f}')
        print(f'  Macro AUC  : {auc:.4f}')
        print(classification_report(labels_np, preds, target_names=GROUP_LABELS, zero_division=0))
        print('  Confusion matrix:')
        print(confusion_matrix(labels_np, preds))

    # Save test predictions
    _, _, _, test_preds, test_probs, test_labels = evaluate(eval_model, te_loader, loss_fn, device)
    np.savez(RESULTS_OUT,
             y_true=test_labels, y_pred=test_preds, y_proba=test_probs,
             enrolid=meta['enrolid'][idx_test])

    total_time = time.time() - load_start
    print('\n' + '='*70)
    print(f'  Data loading  : {timedelta(seconds=int(load_time))}')
    print(f'  Optuna HPO    : {timedelta(seconds=int(optuna_time))}')
    print(f'  Final training: {timedelta(seconds=int(final_train_time))}')
    print(f'  TOTAL         : {timedelta(seconds=int(total_time))}')
    print(f'\nOutputs in {DATA_DIR}:')
    print(f'  {os.path.basename(MODEL_OUT)}')
    print(f'  {os.path.basename(STUDY_OUT)}')
    print(f'  {os.path.basename(RESULTS_OUT)}')


if __name__ == '__main__':
    main()
