"""Re-run XGBoost inference on the MarketScan TEST split.

Loads the trained model (xgboost_4class_model.json) + the one-hot feature CSR,
applies the same stratified split (SEED=35, TEST_FRAC=0.10) as the training
script, and writes predictions in the canonical schema.

Output: outputs/predictions/marketscan_xgb.npz
  y_true (int32), y_pred (int64), y_proba (N, 4 float32), enrolid (int64)
"""

import os
import sys
import time

import numpy as np
import xgboost as xgb
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETSCAN_PRED_OUT

DATA_DIR    = os.environ.get('HIPFX_BASELINES_DIR', 'data/baselines')
FEAT_FILE   = os.path.join(DATA_DIR, 'one_hot_features_4groups.npz')
META_FILE   = os.path.join(DATA_DIR, 'one_hot_labels_metadata_4groups.npz')
MODEL_FILE  = os.path.join(DATA_DIR, 'xgboost_4class_model.json')
OUT_PATH    = MARKETSCAN_PRED_OUT['XGBoost']

# Must match train_xgboost_4class.py
SEED      = 35
TEST_FRAC = 0.10


def main():
    print('Loading data...', flush=True)
    t0 = time.time()
    X = load_npz(FEAT_FILE)
    meta = np.load(META_FILE, allow_pickle=True)
    y = meta['y'].astype(np.int32)
    enrolid = meta['enrolid']
    print(f'  X: {X.shape}  y: {y.shape}  ({time.time()-t0:.1f}s)', flush=True)

    print('\nStratified split (same SEED=35 as training)...', flush=True)
    idx = np.arange(len(y))
    _, idx_test, _, y_test = train_test_split(
        idx, y, test_size=TEST_FRAC, stratify=y, random_state=SEED,
    )
    X_test = X[idx_test]
    enrolid_test = enrolid[idx_test]
    print(f'  TEST n = {len(idx_test):,}', flush=True)

    print(f'\nLoading model: {MODEL_FILE}', flush=True)
    booster = xgb.Booster()
    booster.load_model(MODEL_FILE)

    print('Running inference...', flush=True)
    t0 = time.time()
    dtest = xgb.DMatrix(X_test)
    y_proba = booster.predict(dtest).astype(np.float32)  # (N, 4)
    y_pred  = y_proba.argmax(axis=1).astype(np.int64)
    print(f'  inference: {time.time()-t0:.1f}s', flush=True)

    np.savez(OUT_PATH,
             y_true=y_test, y_pred=y_pred, y_proba=y_proba, enrolid=enrolid_test)
    print(f'\nSaved -> {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
