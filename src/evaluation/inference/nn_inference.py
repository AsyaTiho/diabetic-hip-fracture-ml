"""Re-run NN inference on the MarketScan TEST split.

Loads the trained SparseClassifier weights (nn_4class_weights.pth) and the
one-hot feature CSR, applies the same stratified split (SEED=35, TEST_FRAC=0.10)
as the training script, and writes predictions in the canonical schema.

Output: outputs/predictions/marketscan_nn.npz
  y_true (int32), y_pred (int64), y_proba (N, 4 float32), enrolid (int64)
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz, csr_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETSCAN_PRED_OUT

DATA_DIR     = os.environ.get('HIPFX_BASELINES_DIR', 'data/baselines')
FEAT_FILE    = os.path.join(DATA_DIR, 'one_hot_features_4groups.npz')
META_FILE    = os.path.join(DATA_DIR, 'one_hot_labels_metadata_4groups.npz')
WEIGHTS_FILE = os.path.join(DATA_DIR, 'nn_4class_weights.pth')
OUT_PATH     = MARKETSCAN_PRED_OUT['NN']

SEED       = 35
TEST_FRAC  = 0.10
BATCH_SIZE = 2048
NUM_CLASSES = 4


class SparseClassifier(nn.Module):
    """Identical to train_nn_4class.py:SparseClassifier."""
    def __init__(self, input_size: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SparseIndexDataset(Dataset):
    def __init__(self, indices, y):
        self.indices = indices
        self.y = y
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.indices[idx], self.y[self.indices[idx]]


def make_collate_fn(X: csr_matrix):
    def collate_fn(batch):
        orig_indices, targets = zip(*batch)
        dense = X[list(orig_indices)].toarray().astype(np.float32)
        return torch.from_numpy(dense), torch.tensor(np.array(targets), dtype=torch.long)
    return collate_fn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n', flush=True)

    print('Loading data...', flush=True)
    t0 = time.time()
    X = load_npz(FEAT_FILE)
    meta = np.load(META_FILE, allow_pickle=True)
    y = meta['y'].astype(np.int32)
    enrolid = meta['enrolid']
    print(f'  X: {X.shape}  y: {y.shape}  ({time.time()-t0:.1f}s)', flush=True)

    print('Stratified split (same SEED=35 as training)...', flush=True)
    idx = np.arange(len(y))
    _, idx_test, _, y_test = train_test_split(
        idx, y, test_size=TEST_FRAC, stratify=y, random_state=SEED,
    )
    enrolid_test = enrolid[idx_test]
    print(f'  TEST n = {len(idx_test):,}', flush=True)

    print(f'\nLoading weights: {WEIGHTS_FILE}', flush=True)
    state = torch.load(WEIGHTS_FILE, map_location='cpu', weights_only=False)
    sd = state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state
    model = SparseClassifier(input_size=X.shape[1], num_classes=NUM_CLASSES)
    model.load_state_dict(sd)
    model.to(device).eval()

    loader = DataLoader(
        SparseIndexDataset(idx_test, y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=(device.type == 'cuda'),
        collate_fn=make_collate_fn(X),
    )

    print('\nRunning inference...', flush=True)
    t0 = time.time()
    all_logits = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            all_logits.append(model(X_batch).float().cpu())
    logits = torch.cat(all_logits)
    y_proba = torch.softmax(logits, dim=1).numpy().astype(np.float32)
    y_pred  = logits.argmax(dim=1).numpy().astype(np.int64)
    print(f'  inference: {time.time()-t0:.1f}s', flush=True)

    np.savez(OUT_PATH,
             y_true=y_test, y_pred=y_pred, y_proba=y_proba, enrolid=enrolid_test)
    print(f'\nSaved -> {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
