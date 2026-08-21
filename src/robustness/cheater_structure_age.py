"""Structure-only and age-gap "cheater" pre-flight gate.

Supports the Discussion "Robustness of the risk signal" analysis (the
structure-only / age-gap cheater tests). Two cheaters, both logistic regressions
fitted on the held-out FRACTURE CASES ONLY, scored by inter-fracture macro AUROC
(chance = 0.50):

  SHAPE    sequence length, position of the diabetes token, #visits.
           No diagnosis codes. No age. On the published data: 0.771
  AGE-GAP  last_age - diabetes_age, and nothing else.
           This IS the original leak. On the published data: 0.903

The age-gap cheater is the ONLY place age is used anywhere in this repository, and
it is used solely to DEMONSTRATE the leak that motivated removing age from the
model. A patient's age is deterministically related to the diabetes-to-fracture
interval that defines the label, so an age embedding leaks the outcome; the
shipped model therefore blanks every age token (see finetune_behrt_lora.py).

Under a landmark BOTH cheaters should collapse to ~0.50, because every patient is
cut at the same offset from their own index: length stops counting down to the
outcome, and the age gap becomes a constant.

  both ~0.50    -> the landmark works. Proceed to the fine-tunes.
  shape high    -> the leak has a channel we have not found. STOP.
  age-gap high  -> age is still a live leak channel, which is exactly why the
                   shipped model blanks age.

Writes robustness/cheater_results.csv (these numbers ARE a manuscript panel, so we
do not want to run this twice).

    python cheater_structure_age.py --data $HIPFX_BEHRT_DIR/patient_sequences_finetune.pkl --arm published
    python cheater_structure_age.py --data $HIPFX_BEHRT_DIR/patient_sequences_landmark_30.pkl  --arm landmark_30  --split $HIPFX_BEHRT_DIR/split_landmark_30.npz
    python cheater_structure_age.py --data $HIPFX_BEHRT_DIR/patient_sequences_landmark_180.pkl --arm landmark_180 --split $HIPFX_BEHRT_DIR/split_landmark_180.npz

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
ap.add_argument('--arm', required=True, help='published | landmark_30 | landmark_180')
ap.add_argument('--split', default=None,
                help='split_landmark_*.npz; if omitted, derive the published split')
ap.add_argument('--age-removed', action='store_true',
                help='blank ages so the age-gap channel is 0 (for the age_removed '
                     'arm, whose sequences are the published ones with age blanked)')
args = ap.parse_args()

print(f'=== GATE: {args.arm} ===', flush=True)
print(f'data: {args.data}', flush=True)

df = pd.read_pickle(args.data)
print(f'  {len(df):,} patients', flush=True)

# the held-out test set
if args.split:
    te = np.load(args.split)['test']
    test_df = df[df['enrolid'].isin(set(te.tolist()))].reset_index(drop=True)
    print(f'  test split from {os.path.basename(args.split)}: {len(test_df):,}',
          flush=True)
else:
    _, test_df = train_test_split(df, test_size=TEST_FRAC, random_state=SEED,
                                  stratify=df['risk_class'])
    print(f'  test split derived (seed {SEED}): {len(test_df):,}', flush=True)
del df

# fracture cases only: the inter-fracture axis IS the leak axis
test_df = test_df[test_df['risk_class'].isin([1, 2, 3])].reset_index(drop=True)
print(f'  fracture cases: {len(test_df):,} '
      f'{test_df["risk_class"].value_counts().to_dict()}', flush=True)

# features
L, ev_after, nsep, gap, y = [], [], [], [], []
for codes, ages, rc in zip(test_df['code'], test_df['age'], test_df['risk_class']):
    codes = list(codes)[-MAXLEN:]          # mirror the model's last-512 view
    ages  = list(ages)[-MAXLEN:]
    n = len(codes)
    dm_idx = next((i for i, c in enumerate(codes) if 'iabet' in str(c).lower()), None)
    L.append(n)
    nsep.append(sum(1 for c in codes if c == 'SEP'))
    if dm_idx is None:
        ev_after.append(n)
        gap.append(0.0)
    else:
        ev_after.append(n - dm_idx)
        if args.age_removed:
            gap.append(0.0)                 # ages blanked -> gap is identically 0
        else:
            try:
                gap.append(float(ages[-1]) - float(ages[dm_idx]))
            except Exception:
                gap.append(0.0)
    y.append(rc)

L = np.asarray(L); ev_after = np.asarray(ev_after)
nsep = np.asarray(nsep); gap = np.asarray(gap); y = np.asarray(y)

print(f'\n  seq length : median {np.median(L):.0f}  '
      f'[{np.percentile(L,10):.0f}-{np.percentile(L,90):.0f}]', flush=True)
print(f'  age gap    : median {np.median(gap):.2f} y  '
      f'mean {gap.mean():.2f}  sd {gap.std():.2f}   '
      f'(under a landmark this should be ~0 for everyone)', flush=True)

# run
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []


def run(X, cheater):
    X = X.reshape(len(X), -1) if X.ndim == 1 else X
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
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
shape = run(np.column_stack([np.log1p(L), ev_after, nsep]),
            'shape')       # length + position + #visits. NO age, NO codes
agegap = run(gap, 'age_gap')   # the original leak, in one number

# append to the results CSV
new = pd.DataFrame(rows)
if os.path.exists(OUT_CSV):
    old = pd.read_csv(OUT_CSV)
    old = old[old['arm'] != args.arm]           # replace this arm's rows
    new = pd.concat([old, new], ignore_index=True)
new.to_csv(OUT_CSV, index=False)
print(f'\nwrote {OUT_CSV}', flush=True)

# verdict
print('\n' + '=' * 72, flush=True)
print(f'{args.arm}:  shape {shape:.4f}   age-gap {agegap:.4f}   (chance 0.50)',
      flush=True)
print('\nreference, published data:  shape 0.771   age-gap 0.903', flush=True)
print('reference, full models:     original 0.968   age-removed 0.895', flush=True)
if args.arm != 'published':
    if shape < 0.58 and agegap < 0.58:
        print('\nGATE PASSED. Both cheaters collapsed to near chance.', flush=True)
        print('The landmark kills the leak. Proceed to the fine-tunes.', flush=True)
    else:
        print('\nGATE FAILED.', flush=True)
        if shape >= 0.58:
            print(f'  shape cheater still scores {shape:.3f}: the leak has a '
                  f'channel we have not found.', flush=True)
        if agegap >= 0.58:
            print(f'  age-gap cheater still scores {agegap:.3f}: age is still '
                  f'a live leak channel (the shipped model blanks age for '
                  f'exactly this reason).', flush=True)
        print('  Understand this before running the fine-tunes.', flush=True)
print('=' * 72, flush=True)
