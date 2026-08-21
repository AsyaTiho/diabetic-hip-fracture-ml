"""Step 5 of the matched-cohort pipeline.

Supports the Discussion "Robustness of the risk signal" analysis (matched
cohort). Subsets the existing landmark Delta=180 sequences to the matched k=5
enrolids and emits the two files the fine-tune script consumes:

  matched_sequences_180_k{K}.pkl   same columns as patient_sequences_landmark_180
  split_matched_180_k{K}.npz       train / val / test enrolid arrays

No re-cut and no new split: the landmark sequences already exist and the split is
inherited per-enrolid from the original Delta=180 split (carried in the parquet's
`split` column), so the matched test set stays a strict subset of the original
test set. The landmark sequence pickle is read once; only the matched rows are
kept.

Env vars:
  HIPFX_BEHRT_DIR  base data dir holding patient_sequences_landmark_180.pkl and
                   the matching/ artifacts; the matched sequences pickle and split
                   npz are written here (default data/behrt)
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
M    = os.path.join(DATA_DIR, 'matching')
SEQS = os.path.join(DATA_DIR, 'patient_sequences_landmark_180.pkl')
CLS  = {0: 'Control', 1: '1-5y', 2: '<1y', 3: '5-10y'}

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=5)
args = ap.parse_args()
K = args.k

matched = pd.read_parquet(os.path.join(M, f'matched_cohort_180_k{K}.parquet'))
print(f'matched k={K}: {len(matched):,} patients', flush=True)
want = set(matched['enrolid'].to_numpy().tolist())
split_of = dict(zip(matched['enrolid'].to_numpy(), matched['split'].to_numpy()))

print(f'loading {SEQS} ...', flush=True)
seq = pd.read_pickle(SEQS)
print(f'  {len(seq):,} rows, cols {list(seq.columns)}', flush=True)

sub = seq[seq['enrolid'].isin(want)].reset_index(drop=True)
del seq
print(f'  kept {len(sub):,} of {len(want):,} matched enrolids', flush=True)
assert len(sub) == len(want), \
    f'{len(want) - len(sub):,} matched enrolids missing from the landmark pickle'

# risk_class must agree between the two sources
chk = sub.merge(matched[['enrolid', 'risk_class']], on='enrolid',
                suffixes=('_seq', '_match'))
bad = int((chk['risk_class_seq'] != chk['risk_class_match']).sum())
assert bad == 0, f'{bad} risk_class disagreements between sequences and matched table'
print('  risk_class agrees between sequences and matched table', flush=True)

print('\nclass distribution:', flush=True)
for c in (0, 1, 2, 3):
    print(f'  {c} {CLS[c]:<8} {int((sub.risk_class == c).sum()):>10,}', flush=True)

# split arrays, from the inherited assignment
tr = matched.loc[matched.split == 'train', 'enrolid'].to_numpy(np.int64)
va = matched.loc[matched.split == 'val',   'enrolid'].to_numpy(np.int64)
te = matched.loc[matched.split == 'test',  'enrolid'].to_numpy(np.int64)
print(f'\nsplit: train {len(tr):,}  val {len(va):,}  test {len(te):,}', flush=True)
assert len(tr) + len(va) + len(te) == len(matched), 'split does not partition cohort'
assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te), \
    'split overlap'

out_pkl = os.path.join(DATA_DIR, f'matched_sequences_180_k{K}.pkl')
out_npz = os.path.join(DATA_DIR, f'split_matched_180_k{K}.npz')
sub.to_pickle(out_pkl)
np.savez(out_npz, train=tr, val=va, test=te)
print(f'\nwrote {out_pkl}', flush=True)
print(f'wrote {out_npz}', flush=True)
