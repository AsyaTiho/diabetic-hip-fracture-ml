"""Step 2 (1:k nearest-neighbour variant) of the matched-cohort pipeline.

Supports the Discussion "Robustness of the risk signal" analysis (matched
cohort). The uncapped "keep any control with a case within the caliper" rule
(run_matching.py) turned out to be an OVERLAP test, not a balancing procedure: it
kept 96.25% of controls and left the standardised mean differences essentially
unchanged (dobyr -0.543 -> -0.488, vis_days +0.346 -> +0.378). Keeping every
eligible control guarantees the control distribution stays the control
distribution.

Here each case takes at most k nearest controls, WITHOUT replacement, so
over-represented cells stop contributing surplus controls and the control
distribution is forced towards the case distribution.

Blocking is on (split, sex, dobyr):
  - split  : match inside train/val/test separately, so the test set is balanced
             on its own terms and no control selection depends on another split's
             cases. Split assignment is inherited from the original Delta=180
             split, so the matched test set stays a strict subset of the original.
  - sex    : exact
  - dobyr  : exact
then the nearest available controls by |visibility difference|, capped at the
+/- 30 day caliper.

Cases are processed in a seeded random order so that no case is systematically
advantaged in claiming scarce controls.

Sweeps several k and reports cohort size and balance for each. Writes one parquet
per k plus a summary CSV; nothing is chosen automatically.

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
SEED = 35
K_VALUES = [1, 2, 5, 10, 20]
CLS = {0: 'Control', 1: '1-5y', 2: '<1y', 3: '5-10y'}
# Enrolids with conflicting DEMO rows, excluded rather than collapsed (see
# run_matching.py). Dataset-specific; supply via HIPFX_AMBIGUOUS_ENROLIDS.
AMBIGUOUS_DOBYR = [int(x) for x in
                   os.environ.get('HIPFX_AMBIGUOUS_ENROLIDS', '').split(',') if x.strip()]

print('loading...', flush=True)
coh = pd.read_parquet(os.path.join(M, 'cohort_180.parquet'))
dem = pd.read_csv(os.path.join(M, 'cohort_180_demographics.csv'))
df = coh.merge(dem, on='enrolid', how='left')
assert len(df) == len(coh), 'demographics join changed the row count'
df = df[~df['enrolid'].isin(AMBIGUOUS_DOBYR)].reset_index(drop=True)
assert not df[['sex', 'dobyr', 'total_visibility']].isna().any().any(), \
    'missing demographics; run_matching.py reported none, so this is a new problem'
df['sex'] = df['sex'].astype(np.int8)
df['dobyr'] = df['dobyr'].astype(np.int16)
df['vis_days'] = df['total_visibility'].astype(float) * DAYS_PER_YEAR
print(f'  {len(df):,} patients', flush=True)


def smd(a, b):
    """Standardised mean difference. |SMD| < 0.1 is the usual 'balanced' bar."""
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def report(d, label):
    cs = d[d.risk_class.isin([1, 2, 3])]
    ct = d[d.risk_class == 0]
    row = {'label': label, 'n_total': len(d), 'n_cases': len(cs),
           'n_controls': len(ct),
           'ratio': len(ct) / max(len(cs), 1)}
    for col in ('dobyr', 'vis_days'):
        row[f'smd_{col}'] = smd(cs[col], ct[col])
    row['smd_male'] = smd((cs.sex == 1).astype(float), (ct.sex == 1).astype(float))
    row['max_abs_smd'] = max(abs(row['smd_dobyr']), abs(row['smd_vis_days']),
                             abs(row['smd_male']))
    return row


def match_k(df, k, rng):
    """Return a boolean keep-mask: all matched cases + their selected controls."""
    keep = np.zeros(len(df), dtype=bool)
    is_case = df['risk_class'].isin([1, 2, 3]).to_numpy()
    vis = df['vis_days'].to_numpy()
    n_case_matched = 0

    for _, idx in df.groupby(['split', 'sex', 'dobyr'], sort=False).indices.items():
        idx = np.asarray(idx)
        ci = idx[is_case[idx]]
        ni = idx[~is_case[idx]]
        if len(ci) == 0 or len(ni) == 0:
            continue

        order = np.argsort(vis[ni], kind='stable')
        ni_sorted = ni[order]
        cv = vis[ni_sorted]                    # sorted control visibilities
        used = np.zeros(len(ni_sorted), dtype=bool)

        for c in rng.permutation(ci):          # seeded order, no case privileged
            v = vis[c]
            lo = np.searchsorted(cv, v - CALIPER_DAYS, side='left')
            hi = np.searchsorted(cv, v + CALIPER_DAYS, side='right')
            if lo >= hi:
                continue                       # no control in caliper: unmatched case
            # expand outward from the insertion point, nearest first, skipping used
            p = np.searchsorted(cv, v, side='left')
            left, right = min(p, hi) - 1, min(p, hi)
            taken = 0
            while taken < k and (left >= lo or right < hi):
                dl = v - cv[left] if left >= lo else np.inf
                dr = cv[right] - v if right < hi else np.inf
                if dl <= dr:
                    if not used[left]:
                        used[left] = True
                        keep[ni_sorted[left]] = True
                        taken += 1
                    left -= 1
                else:
                    if not used[right]:
                        used[right] = True
                        keep[ni_sorted[right]] = True
                        taken += 1
                    right += 1
            if taken > 0:
                keep[c] = True
                n_case_matched += 1
    return keep, n_case_matched


rng_master = np.random.default_rng(SEED)
rows = [report(df, 'unmatched (full Delta=180)')]
print(f"\n{'k':>4} {'total':>12} {'cases':>9} {'controls':>12} {'ratio':>7} "
      f"{'smd_dobyr':>10} {'smd_vis':>9} {'smd_male':>9} {'max|SMD|':>9}", flush=True)
r = rows[0]
print(f"{'--':>4} {r['n_total']:>12,} {r['n_cases']:>9,} {r['n_controls']:>12,} "
      f"{r['ratio']:>6.0f}x {r['smd_dobyr']:>+10.3f} {r['smd_vis_days']:>+9.3f} "
      f"{r['smd_male']:>+9.3f} {r['max_abs_smd']:>9.3f}", flush=True)

for k in K_VALUES:
    rng = np.random.default_rng(SEED)
    keep, n_matched = match_k(df, k, rng)
    sub = df[keep].reset_index(drop=True)
    r = report(sub, f'k={k}')
    r['n_cases_unmatched'] = int(df['risk_class'].isin([1, 2, 3]).sum()) - n_matched
    rows.append(r)
    print(f"{k:>4} {r['n_total']:>12,} {r['n_cases']:>9,} {r['n_controls']:>12,} "
          f"{r['ratio']:>6.1f}x {r['smd_dobyr']:>+10.3f} {r['smd_vis_days']:>+9.3f} "
          f"{r['smd_male']:>+9.3f} {r['max_abs_smd']:>9.3f}", flush=True)

    out = sub[['enrolid', 'risk_class', 'split', 'sex', 'dobyr', 'vis_days']]
    out.to_parquet(os.path.join(M, f'matched_cohort_180_k{k}.parquet'), index=False)

    unm = df[df['risk_class'].isin([1, 2, 3]) & ~keep]
    unm[['enrolid', 'risk_class', 'split', 'sex', 'dobyr', 'vis_days']].to_csv(
        os.path.join(M, f'unmatched_cases_180_k{k}.csv'), index=False)

    print('      per class: ' + ', '.join(
        f'{CLS[c]}={int((sub.risk_class == c).sum()):,}' for c in (0, 1, 2, 3)),
        flush=True)
    print('      split: ' + ', '.join(
        f'{s}={int((sub.split == s).sum()):,}' for s in ('train', 'val', 'test')),
        flush=True)

pd.DataFrame(rows).to_csv(os.path.join(M, 'knn_sweep_summary.csv'), index=False)
print(f'\nwrote {os.path.join(M, "knn_sweep_summary.csv")} and one parquet per k', flush=True)
print('|SMD| < 0.1 is the conventional balanced threshold.', flush=True)
