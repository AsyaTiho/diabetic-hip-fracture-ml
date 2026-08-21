"""Step 2 of the matched-cohort pipeline (overlap variant).

Supports the Discussion "Robustness of the risk signal" analysis (matched
cohort). Pure table operation, no model. Exact blocking on (sex, dobyr) plus a
+/- 30 day caliper on total visibility.

  - a CONTROL is kept if at least one case in its (sex, dobyr) block sits within
    30 days of its visibility
  - a CASE is matched if at least one control in its block sits within 30 days
  - unmatched cases are written out and characterized, NOT dropped silently

Controls are uncapped: every control satisfying the rule is kept, then the union
is de-duplicated. This is "demographically supported controls", not 50/50 classes.
(The 1:k nearest-neighbour variant that actually balances the distributions is in
run_matching_knn.py; this overlap variant is kept for the attrition report.)

Split assignment is carried through from the original Delta=180 split, so the
matched test set is a strict subset of the original test set.

Inputs : <matching>/cohort_180.parquet, <matching>/cohort_180_demographics.csv
Outputs: <matching>/matched_cohort_180.parquet, <matching>/unmatched_cases_180.csv

Env vars:
  HIPFX_BEHRT_DIR  base data dir; matching artifacts live under
                   <HIPFX_BEHRT_DIR>/matching/ (default data/behrt)
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
M = os.path.join(DATA_DIR, 'matching')
CALIPER_DAYS = 30.0
DAYS_PER_YEAR = 365.25
CLS = {0: 'Control', 1: '1-5y', 2: '<1y', 3: '5-10y'}

print('loading cohort + demographics...', flush=True)
coh = pd.read_parquet(os.path.join(M, 'cohort_180.parquet'))
dem = pd.read_csv(os.path.join(M, 'cohort_180_demographics.csv'))
print(f'  cohort {len(coh):,}   demographics {len(dem):,}', flush=True)

df = coh.merge(dem, on='enrolid', how='left')
assert len(df) == len(coh), 'demographics join changed the row count (fan-out)'

# Some patients may carry conflicting DEMO rows (e.g. two different birth years
# for the same efamid and sex). Birth year is an exact match key, so guessing is
# not acceptable, and such patients are excluded rather than collapsed with MIN().
# The offending enrolid(s) are dataset-specific data values and are not published
# here; supply them via HIPFX_AMBIGUOUS_ENROLIDS (comma-separated). On the
# licensed MarketScan build this was a single patient, a control in the train
# split, so the cost was nil. If the list grows, revisit the collapse rule.
AMBIGUOUS_DOBYR = [int(x) for x in
                   os.environ.get('HIPFX_AMBIGUOUS_ENROLIDS', '').split(',') if x.strip()]
n_amb = int(df['enrolid'].isin(AMBIGUOUS_DOBYR).sum())
print(f'\nexcluded for conflicting DEMO rows: {n_amb:,}', flush=True)
df = df[~df['enrolid'].isin(AMBIGUOUS_DOBYR)].reset_index(drop=True)

# attrition: patients we cannot match on at all
miss = df[['sex', 'dobyr', 'total_visibility']].isna()
print('\nmissing demographics (cannot be matched or serve as a control):', flush=True)
for c in ('sex', 'dobyr', 'total_visibility'):
    n = int(miss[c].sum())
    print(f'  {c:<18} {n:>10,}  ({100*n/len(df):.4f}%)', flush=True)
bad = miss.any(axis=1)
print(f'  ANY missing        {int(bad.sum()):>10,}  ({100*bad.mean():.4f}%)', flush=True)
if bad.any():
    print('  by class: ' + ', '.join(
        f'{CLS[c]}={int(((df["risk_class"] == c) & bad).sum()):,}' for c in (0, 1, 2, 3)),
        flush=True)

dropped_incomplete = df[bad].copy()
df = df[~bad].reset_index(drop=True)
df['sex'] = df['sex'].astype(np.int8)
df['dobyr'] = df['dobyr'].astype(np.int16)
df['vis_days'] = df['total_visibility'].astype(float) * DAYS_PER_YEAR
print(f'\nmatchable population: {len(df):,}', flush=True)

# the match
is_case = df['risk_class'].isin([1, 2, 3]).to_numpy()
keep = np.zeros(len(df), dtype=bool)

print(f'\nmatching, caliper +/- {CALIPER_DAYS:.0f} days on visibility...', flush=True)
n_blocks = 0
for (sex, dobyr), idx in df.groupby(['sex', 'dobyr'], sort=False).indices.items():
    idx = np.asarray(idx)
    blk_case = is_case[idx]
    ci, ni = idx[blk_case], idx[~blk_case]        # cases, controls in this block
    if len(ci) == 0 or len(ni) == 0:
        continue                                   # block has no possible pairing
    n_blocks += 1
    cv = np.sort(df['vis_days'].to_numpy()[ci])    # sorted case visibilities
    nv = np.sort(df['vis_days'].to_numpy()[ni])

    # a control is kept if SOME case lies in [v-30, v+30]
    v = df['vis_days'].to_numpy()[ni]
    j = np.searchsorted(cv, v - CALIPER_DAYS, side='left')
    ok = (j < len(cv)) & (cv[np.minimum(j, len(cv) - 1)] <= v + CALIPER_DAYS)
    keep[ni] = ok

    # a case is matched if SOME control lies in [v-30, v+30]
    v = df['vis_days'].to_numpy()[ci]
    j = np.searchsorted(nv, v - CALIPER_DAYS, side='left')
    ok = (j < len(nv)) & (nv[np.minimum(j, len(nv) - 1)] <= v + CALIPER_DAYS)
    keep[ci] = ok

print(f'  blocks with both cases and controls: {n_blocks:,}', flush=True)

matched   = df[keep].reset_index(drop=True)
unmatched = df[~keep & is_case].reset_index(drop=True)   # cases only

# report
print('\n=== matched cohort ===', flush=True)
print(f'  total {len(matched):,}', flush=True)
for c in (0, 1, 2, 3):
    n_in  = int((df['risk_class'] == c).sum())
    n_out = int((matched['risk_class'] == c).sum())
    print(f'  {c} {CLS[c]:<8} {n_out:>12,}  of {n_in:>12,}  '
          f'({100*n_out/max(n_in,1):.2f}% kept)', flush=True)
print('\nsplit x class:', flush=True)
print(pd.crosstab(matched['split'], matched['risk_class']).to_string(), flush=True)

print('\n=== unmatched CASES (recorded, not dropped) ===', flush=True)
print(f'  total {len(unmatched):,}', flush=True)
for c in (1, 2, 3):
    n_in  = int(((df['risk_class'] == c)).sum())
    n_out = int((unmatched['risk_class'] == c).sum())
    print(f'  {c} {CLS[c]:<8} {n_out:>8,}  of {n_in:>8,}  '
          f'({100*n_out/max(n_in,1):.2f}% unmatched)', flush=True)

if len(unmatched):
    print('\n  matched vs unmatched cases:', flush=True)
    mc = matched[matched['risk_class'].isin([1, 2, 3])]
    for lab, g in (('matched', mc), ('unmatched', unmatched)):
        print(f'    {lab:<10} n={len(g):>8,}  '
              f'dobyr median {g["dobyr"].median():.0f} '
              f'[{g["dobyr"].quantile(.1):.0f}-{g["dobyr"].quantile(.9):.0f}]  '
              f'vis_days median {g["vis_days"].median():.0f} '
              f'[{g["vis_days"].quantile(.1):.0f}-{g["vis_days"].quantile(.9):.0f}]  '
              f'male {100*(g["sex"] == 1).mean():.1f}%', flush=True)

# QC: the caliper must actually hold
print('\n=== QC ===', flush=True)
mc = matched[matched['risk_class'].isin([1, 2, 3])]
mn = matched[matched['risk_class'] == 0]
print(f'  support: cases span {mc["dobyr"].min()}-{mc["dobyr"].max()}, '
      f'controls span {mn["dobyr"].min()}-{mn["dobyr"].max()}', flush=True)
bad_blocks = 0
for (sex, dobyr), idx in matched.groupby(['sex', 'dobyr'], sort=False).indices.items():
    idx = np.asarray(idx)
    rc = matched['risk_class'].to_numpy()[idx]
    if (rc == 0).all() or (rc != 0).all():
        bad_blocks += 1
print(f'  matched blocks containing only one side: {bad_blocks:,}  (must be 0)',
      flush=True)

out_cols = ['enrolid', 'risk_class', 'split', 'sex', 'dobyr', 'vis_days']
matched[out_cols].to_parquet(os.path.join(M, 'matched_cohort_180.parquet'), index=False)
unmatched[out_cols].to_csv(os.path.join(M, 'unmatched_cases_180.csv'), index=False)
if len(dropped_incomplete):
    dropped_incomplete[['enrolid', 'risk_class', 'split']].to_csv(
        os.path.join(M, 'dropped_incomplete_demographics_180.csv'), index=False)
print(f'\nwrote {os.path.join(M, "matched_cohort_180.parquet")}', flush=True)
print(f'wrote {os.path.join(M, "unmatched_cases_180.csv")}', flush=True)
