"""Diagnostic: what is the residual shape signal actually made of?

Supports the Discussion "Robustness of the risk signal" analysis (the
structure-only cheater tests). The landmark arms leave a shape cheater at ~0.58
(published: 0.771, chance: 0.50). Under a landmark that CANNOT be outcome leakage:
every token the cheater counts occurred at or before index + delta, and the
fracture occurs strictly after it (strict filtration). So the 0.58 must be a
legitimate, available-at-prediction-time signal. This script pins down WHICH one,
instead of arguing about it. No age is used here; only record shape.

Split the same shape features at the index date and fit three models:

  pre-index   tokens and visits BEFORE (and including) the diabetes token.
              The patient's baseline healthcare footprint. Says nothing about
              anything after the index. If this alone carries the 0.58, the
              residual is baseline frailty: sicker patients fracture sooner.

  window      tokens and visits between the index and index + delta.
              Acute activity right after the diabetes diagnosis. If this carries
              it, the residual is deterioration, and it explains why delta=180
              (six times more window to observe) scores above delta=30.

  both        the full shape cheater, restated here so the three are directly
              comparable on the same folds.

Same regression, same folds, same scoring as cheater_structure_age.py, so the
numbers sit alongside the ones already in cheater_results.csv. Appends to that
same CSV: all cheaters end up in one figure.

Read-only. Loads ONE pickle at a time (loading two large sequence pickles at once
is memory-intensive).

Env vars:
  HIPFX_RESULTS_DIR  where robustness/cheater_results.csv is written (default results)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, TEST_FRAC, MAXLEN = 35, 0.10, 512
NAMES = {1: '1-5y', 2: '<1y', 3: '5-10y'}
RESULTS_DIR = os.environ.get('HIPFX_RESULTS_DIR', 'results')
OUT_CSV = os.path.join(RESULTS_DIR, 'robustness', 'cheater_results.csv')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--split', default=None)
args = ap.parse_args()

print(f'=== SHAPE DECOMPOSITION: {args.arm} ===', flush=True)
df = pd.read_pickle(args.data)
print(f'  {len(df):,} patients', flush=True)

if args.split:
    te = set(np.load(args.split)['test'].tolist())
    test_df = df[df['enrolid'].isin(te)].reset_index(drop=True)
else:
    _, test_df = train_test_split(df, test_size=TEST_FRAC, random_state=SEED,
                                  stratify=df['risk_class'])
del df

test_df = test_df[test_df['risk_class'].isin([1, 2, 3])].reset_index(drop=True)
print(f'  fracture cases: {len(test_df):,}', flush=True)

# features, split at the index
pre_len, pre_vis, win_len, win_vis, y = [], [], [], [], []
n_no_dm = 0
for codes, rc in zip(test_df['code'], test_df['risk_class']):
    codes = list(codes)[-MAXLEN:]              # mirror the model's last-512 view
    dm = next((i for i, c in enumerate(codes) if 'iabet' in str(c).lower()), None)
    if dm is None:
        n_no_dm += 1
        dm = len(codes) - 1                    # everything counts as pre-index
    pre, win = codes[:dm + 1], codes[dm + 1:]
    pre_len.append(len(pre))
    pre_vis.append(sum(1 for c in pre if c == 'SEP'))
    win_len.append(len(win))
    win_vis.append(sum(1 for c in win if c == 'SEP'))
    y.append(rc)

pre_len = np.asarray(pre_len); pre_vis = np.asarray(pre_vis)
win_len = np.asarray(win_len); win_vis = np.asarray(win_vis)
y = np.asarray(y)
print(f'  patients with no diabetes token found: {n_no_dm:,}', flush=True)
print(f'  pre-index length: median {np.median(pre_len):.0f}   '
      f'window length: median {np.median(win_len):.0f}', flush=True)

PRE  = np.column_stack([np.log1p(pre_len), np.log1p(pre_vis)])
WIN  = np.column_stack([np.log1p(win_len), np.log1p(win_vis)])
BOTH = np.column_stack([PRE, WIN])

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []


def run(X, cheater):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
    order = clf.fit(X, y).classes_
    proba = proba[:, np.argsort(order)]
    per = {c: roc_auc_score((y == c).astype(int), proba[:, j])
           for j, c in enumerate((1, 2, 3))}
    macro = float(np.mean(list(per.values())))
    print(f'\n{cheater}', flush=True)
    for c in (1, 2, 3):
        print(f'    {NAMES[c]:6s} vs other fractures: {per[c]:.4f}', flush=True)
        rows.append(dict(arm=args.arm, cheater=cheater, cls=NAMES[c],
                         auroc=round(per[c], 4)))
    print(f'    MACRO inter-fracture AUROC: {macro:.4f}   (chance 0.50)', flush=True)
    rows.append(dict(arm=args.arm, cheater=cheater, cls='macro',
                     auroc=round(macro, 4)))
    return macro


print('\n' + '=' * 72, flush=True)
m_pre  = run(PRE,  'shape_preindex')
m_win  = run(WIN,  'shape_window')
m_both = run(BOTH, 'shape_both')

new = pd.DataFrame(rows)
if os.path.exists(OUT_CSV):
    old = pd.read_csv(OUT_CSV)
    old = old[~((old['arm'] == args.arm) &
                (old['cheater'].isin(['shape_preindex', 'shape_window',
                                      'shape_both'])))]
    new = pd.concat([old, new], ignore_index=True)
new.to_csv(OUT_CSV, index=False)

print('\n' + '=' * 72, flush=True)
print(f'{args.arm}:  pre-index {m_pre:.4f} | window {m_win:.4f} | '
      f'both {m_both:.4f}   (chance 0.50)', flush=True)
print(f'wrote {OUT_CSV}', flush=True)
