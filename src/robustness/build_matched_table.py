"""Step 1 of the matched-cohort pipeline.

Supports the Discussion "Robustness of the risk signal" analysis (matched
cohort). Builds the cohort side of the matching table for landmark Delta=180:

    enrolid | risk_class | split

Deliberately does NOT touch the full landmark Delta=180 sequence pickle. The
cohort membership is fully recoverable from split_landmark_180.npz, and
risk_class from hip_frac_labels_CANONICAL.pkl (no sequence columns), which is far
cheaper to load.

The split assignment is carried through rather than regenerated. The matched
cohort must reuse the ORIGINAL Delta=180 split, otherwise the two scoring-only
passes are invalid: a patient who lands in a fresh matched test set but sat in the
original training set has already been seen by the existing checkpoint. Keeping
the split makes the matched test set a strict subset of the original test set.

Writes cohort_180.parquet and cohort_180_enrolids.txt (the latter is the input to
join_demographics.sh, which joins against the proprietary MarketScan demographic
databases).

Env vars:
  HIPFX_BEHRT_DIR  base data dir holding split_landmark_180.npz and
                   hip_frac_labels_CANONICAL.pkl; matching artifacts are written
                   under <HIPFX_BEHRT_DIR>/matching/ (default data/behrt)
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
MATCH_DIR = os.path.join(DATA_DIR, 'matching')
os.makedirs(MATCH_DIR, exist_ok=True)
OUT_PARQUET = os.path.join(MATCH_DIR, 'cohort_180.parquet')
OUT_IDS     = os.path.join(MATCH_DIR, 'cohort_180_enrolids.txt')

CLS = {0: 'Control', 1: '1-5y', 2: '<1y', 3: '5-10y'}

print('loading split...', flush=True)
sp = np.load(os.path.join(DATA_DIR, 'split_landmark_180.npz'))
parts = []
for name in ('train', 'val', 'test'):
    ids = sp[name]
    parts.append(pd.DataFrame({'enrolid': ids, 'split': name}))
    print(f'  {name}: {len(ids):,}', flush=True)
cohort = pd.concat(parts, ignore_index=True)
print(f'  cohort: {len(cohort):,}', flush=True)
assert cohort['enrolid'].is_unique, 'enrolid repeats across splits'

print('\nloading canonical labels (risk_class only)...', flush=True)
lab = pd.read_pickle(os.path.join(DATA_DIR, 'hip_frac_labels_CANONICAL.pkl'))[['enrolid', 'risk_class']]
print(f'  {len(lab):,} labelled patients', flush=True)

cohort = cohort.merge(lab, on='enrolid', how='left')
n_missing = int(cohort['risk_class'].isna().sum())
print(f'  unmatched to a label: {n_missing:,}', flush=True)
assert n_missing == 0, 'cohort enrolids missing from the canonical label file'
cohort['risk_class'] = cohort['risk_class'].astype(np.int8)

print('\nclass distribution:', flush=True)
vc = cohort['risk_class'].value_counts().sort_index()
for c in (0, 1, 2, 3):
    print(f'  {c} {CLS[c]:<8} {int(vc.get(c, 0)):>12,}', flush=True)

# the counts asserted for the study, derived independently in build_landmark_cohort.py
EXPECTED = {0: 11862676, 1: 57592, 2: 15996, 3: 22662}
got = {c: int(vc.get(c, 0)) for c in (0, 1, 2, 3)}
if got != EXPECTED:
    print(f'\nWARNING: class counts differ from the expected study counts.', flush=True)
    print(f'  expected {EXPECTED}', flush=True)
    print(f'  got      {got}', flush=True)
else:
    print('\nclass counts match the study exactly.', flush=True)

print('\nsplit x class:', flush=True)
print(pd.crosstab(cohort['split'], cohort['risk_class']).to_string(), flush=True)

cohort.to_parquet(OUT_PARQUET, index=False)
np.savetxt(OUT_IDS, cohort['enrolid'].to_numpy(), fmt='%d')
print(f'\nwrote {OUT_PARQUET}  ({len(cohort):,} rows)', flush=True)
print(f'wrote {OUT_IDS}', flush=True)
