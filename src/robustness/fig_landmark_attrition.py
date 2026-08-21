"""Landmark attrition under STRICT filtration, for delta = 0, 30, 90, 180 days.

Supports the Discussion "Robustness of the risk signal" analysis (landmark
design): it quantifies what each landmark offset costs in retained patients, so
the choice of delta can be justified.

Landmark rule
-------------
Retain each patient's history from the start of record up to and including
day (diabetes_index + delta). STRICT FILTRATION: any patient whose first hip
fracture falls at or before day (index + delta) is DROPPED from the cohort
entirely. We never truncate a sequence just before a fracture; that asymmetric
truncation is what produced the original leak.

Consequence: the only class that can lose patients at delta <= 180 is class 2
(<1 year), whose fractures live in days 1..365. Classes 1 (1-5y) and 3 (5-10y)
have all fractures past day 365 and are untouched. Class 0 is the control class.

Read-only. Writes one figure + one CSV under the results figs dir.

Env vars:
  HIPFX_BEHRT_DIR    holds hip_frac_labels_CANONICAL.pkl (default data/behrt)
  HIPFX_RESULTS_DIR  where figs/ output is written (default results)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
RESULTS_DIR = os.environ.get('HIPFX_RESULTS_DIR', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
LABELS = os.path.join(DATA_DIR, 'hip_frac_labels_CANONICAL.pkl')
DELTAS = [0, 30, 90, 180]
CLASSES = {0: 'Control (>10y / none)', 1: '1-5 years', 2: '<1 year', 3: '5-10 years'}

print('loading', LABELS, flush=True)
df = pd.read_pickle(LABELS)
print(f'  {len(df):,} patients; cols: {df.columns.tolist()}', flush=True)

dt = df['days_to_hip_frac'].values.astype('float64')   # NaN where no fracture
rc = df['risk_class'].values.astype(int)

base = pd.Series(rc).value_counts().sort_index()
print('\nbaseline class counts:', flush=True)
for c in [0, 1, 2, 3]:
    print(f'  {c} {CLASSES[c]:<24} {int(base.get(c,0)):>12,}', flush=True)

# Sanity: fractures that precede or coincide with the index (dt <= 0). These
# patients are already class 0, but their fracture sits BEFORE the landmark
# window, so no delta can make their record fracture-free. Count them.
pre_index = int(np.nansum(dt <= 0))
print(f'\nfracture at/before index (dt<=0, already class 0): {pre_index:,}', flush=True)
print('  -> these have a fracture in the retained window at ANY delta;', flush=True)
print('     strict filtration drops them regardless of delta.', flush=True)

# ---- attrition table ------------------------------------------------------
rows = []
for d in DELTAS:
    # dropped = any patient whose fracture is at or before index+delta.
    # includes dt <= 0 (pre-index fractures) at every delta.
    drop = np.zeros(len(df), dtype=bool)
    with np.errstate(invalid='ignore'):
        drop |= (dt <= d) & ~np.isnan(dt)
    for c in [0, 1, 2, 3]:
        m = rc == c
        n0 = int(m.sum())
        nd = int((m & drop).sum())
        rows.append(dict(delta=d, risk_class=c, name=CLASSES[c],
                         n_before=n0, n_dropped=nd, n_after=n0 - nd,
                         pct_dropped=100 * nd / n0 if n0 else 0.0))
tab = pd.DataFrame(rows)

print('\n' + '=' * 88, flush=True)
print('STRICT-FILTRATION ATTRITION', flush=True)
print('=' * 88, flush=True)
for d in DELTAS:
    s = tab[tab.delta == d]
    tot_b = s.n_before.sum(); tot_d = s.n_dropped.sum()
    print(f'\ndelta = {d:>3} days   (retain history through index+{d}d)', flush=True)
    print(f'  {"class":<24}{"before":>12}{"dropped":>11}{"after":>12}{"lost":>9}', flush=True)
    for _, r in s.iterrows():
        print(f'  {r["name"]:<24}{r.n_before:>12,}{r.n_dropped:>11,}'
              f'{r.n_after:>12,}{r.pct_dropped:>8.2f}%', flush=True)
    print(f'  {"TOTAL":<24}{tot_b:>12,}{tot_d:>11,}{tot_b-tot_d:>12,}'
          f'{100*tot_d/tot_b:>8.2f}%', flush=True)

tab.to_csv(os.path.join(FIGS_DIR, 'landmark_delta_attrition.csv'), index=False)

# ---- the <1y class in detail ---------------------------------------------
c2 = dt[(rc == 2)]
c2 = c2[~np.isnan(c2)]
print('\n' + '=' * 88, flush=True)
print(f'<1 YEAR CLASS: distribution of days from index to fracture (N={len(c2):,})', flush=True)
print('=' * 88, flush=True)
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f'  p{p:<3} = {np.percentile(c2, p):>5.0f} days', flush=True)
print(f'  fractures in first 30d : {int((c2<=30).sum()):>7,} ({100*(c2<=30).mean():.1f}%)', flush=True)
print(f'  fractures in first 90d : {int((c2<=90).sum()):>7,} ({100*(c2<=90).mean():.1f}%)', flush=True)
print(f'  fractures in first 180d: {int((c2<=180).sum()):>7,} ({100*(c2<=180).mean():.1f}%)', flush=True)

# ---- figure ---------------------------------------------------------------
OK = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']   # Okabe-Ito
fig = plt.figure(figsize=(13.5, 4.4))
gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.06, right=0.985, top=0.86, bottom=0.16)

# panel A: cumulative attrition curve for the <1y class
axA = fig.add_subplot(gs[0, 0])
xs = np.arange(0, 366)
ys = np.array([100 * (c2 <= x).mean() for x in xs])
axA.plot(xs, ys, color=OK[0], lw=2.2, zorder=3)
for i, d in enumerate(DELTAS):
    y = 100 * (c2 <= d).mean()
    axA.plot([d, d], [0, y], color=OK[1], ls='--', lw=1.1, zorder=2)
    axA.plot([d], [y], 'o', color=OK[1], ms=6, zorder=4)
    axA.annotate(f'{y:.0f}%', (d, y), textcoords='offset points',
                 xytext=(6, -2), fontsize=9, color=OK[1], fontweight='bold')
axA.set_xlabel('landmark $\\Delta$ (days after diabetes index)')
axA.set_ylabel('% of <1y class dropped')
axA.set_title('A. Attrition of the <1 year class', fontsize=11, loc='left', fontweight='bold')
axA.set_xlim(0, 365); axA.set_ylim(0, 100)
axA.grid(alpha=.25, lw=.6)

# panel B: surviving N per class, grouped bars over delta
axB = fig.add_subplot(gs[0, 1])
w = 0.2
xpos = np.arange(3)                      # fracture classes only (0 never drops here)
order = [2, 1, 3]
for i, d in enumerate(DELTAS):
    vals = [int(tab[(tab.delta == d) & (tab.risk_class == c)].n_after.iloc[0]) for c in order]
    b = axB.bar(xpos + (i - 1.5) * w, vals, w, label=f'$\\Delta$={d}d',
                color=OK[i], edgecolor='white', lw=.6)
    if i == 0:
        continue
axB.set_xticks(xpos)
axB.set_xticklabels([CLASSES[c] for c in order])
axB.set_ylabel('surviving patients')
axB.set_title('B. Fracture cases retained', fontsize=11, loc='left', fontweight='bold')
axB.legend(frameon=False, fontsize=8.5, ncol=2)
axB.grid(alpha=.25, lw=.6, axis='y')
axB.ticklabel_format(axis='y', style='plain')

# panel C: histogram of time-to-fracture in the <1y class with delta cuts
axC = fig.add_subplot(gs[0, 2])
axC.hist(c2, bins=np.arange(0, 371, 10), color='#BBBBBB', edgecolor='white', lw=.5)
ymax = axC.get_ylim()[1]
for i, d in enumerate(DELTAS[1:], start=1):
    axC.axvline(d, color=OK[i], lw=1.8, ls='--',
                label=f'$\\Delta$={d}d  ({100*(c2<=d).mean():.0f}% lost)')
axC.set_xlabel('days from diabetes index to hip fracture')
axC.set_ylabel('patients')
axC.set_title('C. When the <1y fractures happen', fontsize=11, loc='left', fontweight='bold')
axC.legend(frameon=False, fontsize=8.5)
axC.grid(alpha=.25, lw=.6, axis='y')
axC.set_xlim(0, 365)

fig.suptitle('Index-date landmark with strict filtration: what each $\\Delta$ costs',
             fontsize=12.5, fontweight='bold', x=0.06, ha='left')
out = os.path.join(FIGS_DIR, 'landmark_delta_attrition.png')
fig.savefig(out, dpi=170)
print(f'\nfigure -> {out}', flush=True)
print(f'table  -> {os.path.join(FIGS_DIR, "landmark_delta_attrition.csv")}', flush=True)
print('DONE.', flush=True)
