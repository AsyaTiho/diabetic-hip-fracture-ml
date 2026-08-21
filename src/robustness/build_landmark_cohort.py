"""Build the landmark fine-tune dataset.

Supports the Discussion "Robustness of the risk signal" analysis (landmark
design). Replaces the leaky truncation used by the primary fine-tune build:
build_finetune_dataset.py cuts each sequence strictly before the first
hip-fracture token, so the sequence ends AT THE OUTCOME and its length counts
down to the label. That is the leak.

Here every patient is cut at their own `diabetes_day + delta`. Because
patient_sequences_dayed.pkl carries a real `day` on every token, the cut is an
exact comparison, not an inference.

    python build_landmark_cohort.py --delta 30
    python build_landmark_cohort.py --delta 180

Patients are dropped for exactly three reasons, each counted and reported:
  1. no diabetes index  (diabetes_day is NaN, no landmark can be computed)
  2. fracture at or before the cutoff  (days_to_hip_frac <= delta). This covers
     BOTH the in-window fractures AND the 53,238 whose fracture PRECEDES their
     diabetes index (days_to_hip_frac <= 0), which rc() currently sweeps into the
     control class. STRICT FILTRATION: we delete them, we do not truncate them.
     Truncating is what created the leak.
  3. empty sequence  (no event on or before the cutoff)

Two assertions must hold, or the script exits non-zero:
  A. ZERO surviving hip-fracture tokens. This is provable (MS_745.11/MS_745.9 map
     one-to-one onto 'Fracture of femur'/'Pathological fracture', and we drop
     everyone who fractures by the cutoff), so it can only fire if the day stamps,
     the label file, and the tokens disagree, i.e. the exact silent failure we
     care about.
  B. Surviving class counts match the counts derived INDEPENDENTLY from the label
     file alone, before any sequence was touched.

Output: patient_sequences_landmark_{delta}.pkl (enrolid | code | age | day | risk_class)
Nothing existing is modified or overwritten.

Env vars:
  HIPFX_BEHRT_DIR  base data dir holding patient_sequences_dayed.pkl and
                   hip_frac_labels_CANONICAL.pkl, and where the landmark pickle is
                   written (default data/behrt)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
SEQS   = os.path.join(DATA_DIR, 'patient_sequences_dayed.pkl')
LABELS = os.path.join(DATA_DIR, 'hip_frac_labels_CANONICAL.pkl')

HIP_FRACTURE_PHENOTYPES = {'Fracture of femur', 'Pathological fracture'}
CLS = {0: 'Control (>10y / none)', 1: '1-5 years', 2: '<1 year', 3: '5-10 years'}

ap = argparse.ArgumentParser()
ap.add_argument('--delta', type=int, required=True, help='landmark, days after index')
args = ap.parse_args()
D = args.delta
OUT = os.path.join(DATA_DIR, f'patient_sequences_landmark_{D}.pkl')

t0 = time.time()
print(f'=== LANDMARK delta = {D} days ===\n', flush=True)

# load
print('loading day-stamped sequences...', flush=True)
seq = pd.read_pickle(SEQS)
print(f'  {len(seq):,} patients, cols {seq.columns.tolist()}', flush=True)
assert 'day' in seq.columns, 'sequences have no day column; run the rebuild first'

print('loading labels...', flush=True)
lab = pd.read_pickle(LABELS)[['enrolid', 'diabetes_day', 'days_to_hip_frac',
                              'risk_class']]
seq = seq.merge(lab, on='enrolid', how='left')
n_start = len(seq)
print(f'  merged; {n_start:,} patients\n', flush=True)

# the independent expectation (label file alone, no sequences)
dt_l = lab['days_to_hip_frac'].values.astype('float64')
rc_l = lab['risk_class'].values.astype(int)
keep_l = ~((dt_l <= D) & ~np.isnan(dt_l))
expected = {c: int(((rc_l == c) & keep_l).sum()) for c in (0, 1, 2, 3)}
print('expected surviving counts (from the LABEL FILE alone):', flush=True)
for c in (2, 1, 3, 0):
    print(f'  {c} {CLS[c]:<22} {expected[c]:>12,}', flush=True)
print('', flush=True)

# DROP 1: no diabetes index
no_dm = seq['diabetes_day'].isna()
n_no_dm = int(no_dm.sum())
print(f'DROP 1  no diabetes index (diabetes_day is NaN): {n_no_dm:,} '
      f'({100*n_no_dm/n_start:.3f}%)', flush=True)
if n_no_dm:
    print('   by class: ' + ', '.join(
        f'{CLS[c]}={int((seq.loc[no_dm, "risk_class"] == c).sum()):,}'
        for c in (0, 1, 2, 3)), flush=True)
seq = seq[~no_dm]

# DROP 2: strict filtration
dt = seq['days_to_hip_frac'].values.astype('float64')
frac_in_window = (dt <= D) & ~np.isnan(dt)
n_pre_index = int(((dt <= 0) & ~np.isnan(dt)).sum())
n_in_window = int(((dt > 0) & (dt <= D)).sum())
print(f'\nDROP 2  strict filtration: {int(frac_in_window.sum()):,}', flush=True)
print(f'   fracture BEFORE/AT the diabetes index (dt <= 0) : {n_pre_index:,}',
      flush=True)
print(f'   fracture inside the window (0 < dt <= {D})       : {n_in_window:,}',
      flush=True)
seq = seq[~frac_in_window]

# THE CUT
print(f'\ncutting every sequence at diabetes_day + {D} ...', flush=True)
cutoff = (seq['diabetes_day'].values + D).astype(np.int64)

codes, ages, days, n_keep = [], [], [], []
for c, a, dd, cut in zip(seq['code'].values, seq['age'].values,
                         seq['day'].values, cutoff):
    dd = np.asarray(dd)
    k = int(np.searchsorted(dd, cut, side='right'))   # day lists are sorted
    n_keep.append(k)
    codes.append(list(c[:k]))
    ages.append(list(a[:k]))
    days.append(list(dd[:k]))

seq = seq.assign(code=codes, age=ages, day=days)
n_keep = np.asarray(n_keep)
print(f'  kept a median of {np.median(n_keep):.0f} tokens per patient '
      f'(was {np.median([len(c) for c in seq["code"]]) if False else "-"})',
      flush=True)

# DROP 3: empty sequences
empty = n_keep == 0
n_empty = int(empty.sum())
print(f'\nDROP 3  empty after the cut (no event on/before the cutoff): {n_empty:,}',
      flush=True)
if n_empty:
    print('   by class: ' + ', '.join(
        f'{CLS[c]}={int((seq.loc[empty, "risk_class"] == c).sum()):,}'
        for c in (0, 1, 2, 3)), flush=True)
seq = seq[~empty].reset_index(drop=True)

# ASSERTION A: no fracture token can survive
print('\nASSERT A: zero surviving hip-fracture tokens...', flush=True)
leaked = sum(1 for c in seq['code'].values
             if any(t in HIP_FRACTURE_PHENOTYPES for t in c))
print(f'  surviving fracture tokens: {leaked:,}  (must be 0)', flush=True)

# ASSERTION B: class counts match the label-file derivation
print('\nASSERT B: surviving class counts vs the independent expectation...',
      flush=True)
got = seq['risk_class'].value_counts().to_dict()
ok_counts = True
print(f'  {"class":<24}{"expected":>12}{"got":>12}{"diff":>10}', flush=True)
for c in (2, 1, 3, 0):
    e, g = expected[c], int(got.get(c, 0))
    d_ = g - e
    print(f'  {CLS[c]:<24}{e:>12,}{g:>12,}{d_:>+10,}', flush=True)
    # class 0 legitimately differs by the DROP-1 and DROP-3 counts
    if c != 0 and d_ != 0:
        ok_counts = False

# sequence length stats
L = np.asarray([len(c) for c in seq['code'].values])
V = np.asarray([sum(1 for t in c if t == 'SEP') for c in seq['code'].values])
print(f'\nretained sequence length (tokens incl. SEP):', flush=True)
print(f'  median {np.median(L):.0f}  mean {L.mean():.1f}  '
      f'p10 {np.percentile(L,10):.0f}  p90 {np.percentile(L,90):.0f}  '
      f'max {L.max():,}', flush=True)
print(f'  fraction under 256 tokens: {100*(L < 256).mean():.1f}%  '
      f'(if very high, MAX_LEN could drop from 512 to 256)', flush=True)
print(f'visits retained: median {np.median(V):.0f}', flush=True)
print('\nby class:', flush=True)
for c in (2, 1, 3, 0):
    m = seq['risk_class'].values == c
    if m.sum():
        print(f'  {CLS[c]:<24} n={int(m.sum()):>12,}  median len '
              f'{np.median(L[m]):>5.0f}  median visits {np.median(V[m]):>4.0f}',
              flush=True)

# verdict
print('\n' + '=' * 72, flush=True)
if leaked != 0:
    print(f'FAILED (A): {leaked:,} hip-fracture tokens survived the cut.', flush=True)
    print('This is provably impossible unless the day stamps, the label file,', flush=True)
    print('and the tokens disagree. STOP: the day stamps are suspect.', flush=True)
    sys.exit(1)
if not ok_counts:
    print('FAILED (B): surviving fracture-class counts do not match the', flush=True)
    print('independent label-file derivation. The cut is off. STOP.', flush=True)
    sys.exit(1)

seq = seq[['enrolid', 'code', 'age', 'day', 'risk_class']]
seq.to_pickle(OUT)
print(f'PASSED. {len(seq):,} patients written to', flush=True)
print(f'  {OUT}', flush=True)
print(f'\ntotal dropped: {n_start - len(seq):,} '
      f'({n_no_dm:,} no-index + {int(frac_in_window.sum()):,} filtered + '
      f'{n_empty:,} empty)', flush=True)
print(f'done in {(time.time()-t0)/60:.1f} min', flush=True)
print('=' * 72, flush=True)
