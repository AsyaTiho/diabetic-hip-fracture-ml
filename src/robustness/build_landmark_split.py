"""Build the landmark train/val/test split.

Supports the Discussion "Robustness of the risk signal" analysis (landmark
design). The landmark arms re-fine-tune the model on histories cut at a fixed
offset from the diabetes index instead of just before the fracture, so the split
must line up patient-for-patient with the primary (published) arm.

    python build_landmark_split.py --delta 30

Why this exists. finetune_behrt_lora.py DERIVES its split internally with
train_test_split(random_state=35) over the fine-tune dataframe, then asserts the
test set equals test_enrolids_2026-06-08.npy. On a strict-filtered landmark cohort
that assert fails, and re-deriving the split would silently hand us a DIFFERENT
test set, which would make the landmark numbers incomparable, patient-for-patient,
to the published and age-removed arms.

So instead: reproduce the PUBLISHED split exactly (and verify it against
test_enrolids_2026-06-08.npy), then SUBTRACT the patients the landmark filtered
out. The landmark test set is then a strict SUBSET of the published test set, and
all four arms can be scored on the same people.

On the 2,717 patients the published fine-tune dataset lacks. The sequence file has
12,033,245 patients; the published fine-tune dataset has 12,030,528, because
truncate_before_hip() emptied 2,717 of them. Those have no published split
assignment. They cannot survive the landmark cut, and this is provable:
  - a patient is emptied only if their FIRST token is a fracture;
  - if the fracture is first, the diabetes token comes later, so
    hip_frac_day <= diabetes_day, so days_to_hip_frac <= 0;
  - so rc() labels them class 0, and strict filtration drops every patient with
    days_to_hip_frac <= delta, which includes <= 0, at any delta.
The data agrees exactly: all 2,717 missing patients are class 0, and the three
fracture-class counts (57,592 / 37,077 / 22,662) are identical in both files.

So the check below is an ASSERTION, not a filter. If it ever finds an orphan, the
day stamps, the label file, and the sequences disagree, and we want to hear about
that loudly rather than let it quietly reshape the test set.

Output: split_landmark_{delta}.npz with train / val / test enrolid arrays.
Nothing is modified or overwritten.

Env vars:
  HIPFX_BEHRT_DIR  base data dir holding patient_sequences_finetune.pkl,
                   test_enrolids_2026-06-08.npy, hip_frac_labels_CANONICAL.pkl,
                   and where split_landmark_{delta}.npz is written (default data/behrt)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
PUBLISHED_FT  = os.path.join(DATA_DIR, 'patient_sequences_finetune.pkl')
TEST_ENROLIDS = os.path.join(DATA_DIR, 'test_enrolids_2026-06-08.npy')
LABELS        = os.path.join(DATA_DIR, 'hip_frac_labels_CANONICAL.pkl')

# Survivor counts printed by build_landmark_cohort.py, which we must reproduce.
EXPECTED = {30: 11_974_721, 180: 11_958_926}

SEED, TEST_FRAC, VAL_FRAC = 35, 0.10, 0.10   # must match finetune_behrt_lora.py

ap = argparse.ArgumentParser()
ap.add_argument('--delta', type=int, required=True)
args = ap.parse_args()
D = args.delta
LANDMARK = os.path.join(DATA_DIR, f'patient_sequences_landmark_{D}.pkl')
OUT      = os.path.join(DATA_DIR, f'split_landmark_{D}.npz')

# 1. reproduce the PUBLISHED split, exactly as finetune_behrt_lora.py does
print('loading the published fine-tune dataset...', flush=True)
pub = pd.read_pickle(PUBLISHED_FT)
print(f'  {len(pub):,} patients', flush=True)

trainval_df, test_df = train_test_split(
    pub, test_size=TEST_FRAC, random_state=SEED, stratify=pub['risk_class'])
train_df, val_df = train_test_split(
    trainval_df, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=SEED,
    stratify=trainval_df['risk_class'])

pub_train = train_df['enrolid'].values.astype(np.int64)
pub_val   = val_df['enrolid'].values.astype(np.int64)
pub_test  = test_df['enrolid'].values.astype(np.int64)
print(f'  published split: train {len(pub_train):,} | val {len(pub_val):,} '
      f'| test {len(pub_test):,}', flush=True)

# 2. VERIFY it against the published test enrolids
ref = np.load(TEST_ENROLIDS)
if not np.array_equal(np.sort(ref), np.sort(pub_test)):
    print('\nFAILED: the reproduced test split does NOT match '
          'test_enrolids_2026-06-08.npy.', flush=True)
    print(f'  reference {len(ref):,} vs reproduced {len(pub_test):,}', flush=True)
    print('STOP: without this, no landmark number is comparable to the '
          'published one.', flush=True)
    sys.exit(1)
print('  VERIFIED: reproduces test_enrolids_2026-06-08.npy exactly.', flush=True)

del pub, trainval_df, train_df, val_df, test_df

# 3. which patients survived the landmark cut? From the LABEL FILE, not the pickle.
#
# Loading the full landmark sequence pickle here is memory-intensive: the
# sequences are Python lists of str objects and blow up in RAM. We do not need
# them. Survival is decided by three rules, and all three are answerable from one
# number per patient:
#   1. no diabetes index      -> dropped   (diabetes_day is NaN; count was 0)
#   2. strict filtration      -> dropped   (days_to_hip_frac <= delta)
#   3. empty after the cut     -> dropped  (count was 0 for BOTH deltas, per the
#                                           build logs, so it cannot bite here)
# so:  survivor <=> diabetes_day present AND NOT (days_to_hip_frac <= delta).
#
# This is the same derivation build_landmark_cohort.py checked its own output
# against (its ASSERT B), and it reproduces the built cohort sizes exactly:
#   delta=30  -> 11,974,721      delta=180 -> 11,958,926
print(f'\nderiving the surviving cohort from the labels (delta = {D})...', flush=True)
lab = pd.read_pickle(LABELS)
surv = lab['diabetes_day'].notna() & ~(lab['days_to_hip_frac'] <= D)
lm_ids = lab.loc[surv, 'enrolid'].values.astype(np.int64)
print(f'  {len(lm_ids):,} patients survived the cut', flush=True)

if len(lm_ids) != EXPECTED[D]:
    print(f'\nFAILED: expected {EXPECTED[D]:,} survivors (the count '
          f'build_landmark_cohort.py wrote), got {len(lm_ids):,}.', flush=True)
    print('The label file and the built dataset disagree. STOP.', flush=True)
    sys.exit(1)
print(f'  matches the built dataset exactly ({EXPECTED[D]:,}).', flush=True)
del lab, surv

pub_all = np.concatenate([pub_train, pub_val, pub_test])
orphan = np.setdiff1d(lm_ids, pub_all)
print(f'  patients with no published split assignment: {len(orphan):,}  '
      f'(must be 0)', flush=True)
if len(orphan):
    print('\nFAILED: a landmark patient survived the cut but has no published', flush=True)
    print('split assignment. That is provably impossible (see the docstring):', flush=True)
    print('every patient emptied by truncate_before_hip has days_to_hip_frac <= 0', flush=True)
    print('and is removed by strict filtration. So the day stamps, the label file,', flush=True)
    print('and the sequences disagree. STOP.', flush=True)
    print(f'  e.g. enrolids {orphan[:5].tolist()}', flush=True)
    sys.exit(1)

# 4. subtract
keep = np.isin(pub_train, lm_ids)
tr = pub_train[keep]
va = pub_val[np.isin(pub_val, lm_ids)]
te = pub_test[np.isin(pub_test, lm_ids)]

print(f'\n{"split":<8}{"published":>14}{"landmark":>12}{"dropped":>10}{"kept %":>9}',
      flush=True)
for nm, p, l in (('train', pub_train, tr), ('val', pub_val, va),
                 ('test', pub_test, te)):
    print(f'{nm:<8}{len(p):>14,}{len(l):>12,}{len(p)-len(l):>10,}'
          f'{100*len(l)/len(p):>8.2f}%', flush=True)

assert np.all(np.isin(te, pub_test)), 'test set is not a subset of the published one'
assert len(np.intersect1d(tr, te)) == 0, 'train/test overlap'
assert len(np.intersect1d(va, te)) == 0, 'val/test overlap'
assert len(np.intersect1d(tr, va)) == 0, 'train/val overlap'
print('\nasserts passed: no overlap; test is a strict subset of the published test.',
      flush=True)

np.savez(OUT, train=tr, val=va, test=te)
print(f'\nwrote {OUT}', flush=True)
print(f'  use with:  BEHRT_SPLIT_FILE={OUT}', flush=True)
