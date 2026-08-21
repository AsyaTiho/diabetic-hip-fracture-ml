"""Discussion composite figure: robustness of the risk signal.

Supports the Discussion "Robustness of the risk signal" analysis, drawing all
three strands (landmark design, matched cohort, and the structure-only / age-gap
cheater tests) into one four-panel figure with one caption:

  a  landmark window   (schematic)   record cut at the fracture vs fixed window
  b  matching          (schematic)   exact block on sex+birth-year, enrollment caliper
  c  shortcut predictors by design   age-gap and record-shape cheaters, cut-at-fracture
                                       vs landmark (chance 0.50)
  d  fracture-timing discrimination   per-class + macro, cut-at-fracture vs landmark
                                       vs matched

Numbers for panels c and d are read at run time from cheater_results.csv and the
per-arm prediction files, so they cannot drift. Shared figure style comes from
src/evaluation/figstyle.py.

Env vars:
  HIPFX_RESULTS_DIR  holds robustness/cheater_results.csv and the per-arm
                     prediction files under outputs/predictions/ (the age-blind
                     primary marketscan_behrt.npz plus the landmark and matched
                     arm .npz written by run_pipeline.py --score), and where
                     figs/ output is written (default results)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.metrics import roc_auc_score

# shared figure style lives in src/evaluation/figstyle.py
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation'))
import figstyle as F
F.apply()

RESULTS_DIR = os.environ.get('HIPFX_RESULTS_DIR', 'results')
PRED_DIR = os.path.join(RESULTS_DIR, 'outputs', 'predictions')
LEAK     = os.path.join(RESULTS_DIR, 'robustness', 'cheater_results.csv')
AGEBLIND = os.path.join(PRED_DIR, 'marketscan_behrt.npz')
LM180    = os.path.join(PRED_DIR, 'landmark_180_behrt.npz')
MATCH    = os.path.join(PRED_DIR, 'matched_180_k5_behrt.npz')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
INK, MUT, GREY = F.INK, F.MUT, F.GREY
BLUE, ORANGE, GREEN = F.BEHRT, F.FRAX, F.XGB
NAMES = ['<1y', '1-5y', '5-10y']

# data for c
ch = pd.read_csv(LEAK)
def cheat(arm, kind):
    r = ch[(ch.arm == arm) & (ch.cheater == kind) & (ch.cls == 'macro')]
    return float(r.auroc.iloc[0])
c_data = {  # cheater -> (cut-at-fracture, landmark)
    'age gap':      (cheat('published', 'age_gap'), cheat('landmark_180', 'age_gap')),
    'record shape': (cheat('published', 'shape'),   cheat('landmark_180', 'shape')),
}

# data for d
def _frac_auroc(npz_path):
    """Per-class fracture-timing AUROC (fracture cases only) from an arm's
    prediction file, keyed by class label for panel d."""
    a = np.load(npz_path, allow_pickle=True)
    ytA = a['y_true'].astype(int); ypA = a['y_proba']
    mm = np.isin(ytA, [1, 2, 3]); yt3A = ytA[mm]
    yp3A = ypA[mm][:, [1, 2, 3]]; yp3A = yp3A / yp3A.sum(1, keepdims=True)
    per = {c: roc_auc_score((yt3A == c).astype(int), yp3A[:, j])
           for j, c in enumerate([1, 2, 3])}
    return {'<1y': per[2], '1-5y': per[1], '5-10y': per[3]}

d = np.load(AGEBLIND, allow_pickle=True)
yt, yp = d['y_true'].astype(int), d['y_proba']
m = np.isin(yt, [1, 2, 3]); yt3 = yt[m]; yp3 = yp[m][:, [1, 2, 3]]; yp3 = yp3 / yp3.sum(1, keepdims=True)
cut_frac = {c: roc_auc_score((yt3 == c).astype(int), yp3[:, j]) for j, c in enumerate([1, 2, 3])}
lm = _frac_auroc(LM180)
mt = _frac_auroc(MATCH)
# order columns as <1y,1-5y,5-10y
d_designs = [
    ('cut at fracture', [cut_frac[2], cut_frac[1], cut_frac[3]], ORANGE),
    ('landmark',        [lm['<1y'], lm['1-5y'], lm['5-10y']],    BLUE),
    ('matched',         [mt['<1y'], mt['1-5y'], mt['5-10y']],    GREEN),
]

# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.2, 6.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.30,
                      left=0.08, right=0.97, top=0.93, bottom=0.09)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

def panel(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', color=INK, ha='right', va='bottom')

# a: landmark window schematic
panel(ax_a, 'a')
ax_a.set_title('landmark window', fontsize=9, color=INK, pad=6)
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 3.4); ax_a.axis('off')
DX, LMX = 2.0, 4.2                       # diabetes index x, landmark cut x
frac_x = [5.0, 6.6, 8.6]                 # fractures at different times (3 patients)
for i, fx in enumerate(frac_x):
    y = 2.7 - i * 0.9
    ax_a.plot([0.6, fx + 0.6], [y, y], color=GREY, lw=0.8, zorder=1)   # full record
    ax_a.add_patch(Rectangle((0.6, y - 0.12), LMX - 0.6, 0.24, facecolor=BLUE,
                   alpha=0.85, edgecolor='none', zorder=2))            # kept window
    ax_a.scatter(DX, y, s=16, color=INK, zorder=4)                      # diabetes index
    ax_a.scatter(fx + 0.6, y, s=42, marker='X', color=ORANGE, zorder=4)  # fracture
ax_a.plot([LMX, LMX], [0.5, 3.0], color=INK, lw=1.1, ls=(0, (3, 2)), zorder=3)
ax_a.text(LMX, 3.02, 'landmark cut\n(diabetes + $\\Delta$)', fontsize=6.6, color=INK,
          ha='center', va='bottom', linespacing=1.2)
ax_a.text(DX, 0.28, 'diabetes', fontsize=6.4, color=MUT, ha='center', va='top')
ax_a.text(8.0, 0.28, 'fracture (varies)', fontsize=6.4, color=ORANGE, ha='center', va='top')
ax_a.text(0.6, -0.02, 'kept: same window for every patient', fontsize=6.4, color=BLUE,
          ha='left', va='top')

# b: matching schematic
panel(ax_b, 'b')
ax_b.set_title('demographic matching', fontsize=9, color=INK, pad=6)
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis('off')
# exact block: sex x birth-year grid, one cell highlighted
YEARS = ['1954', '1955', '1956']
cw, chh, x0 = 0.085, 0.16, 0.06
rows = {'♀': 0.62, '♂': 0.44}
for sym, ry in rows.items():
    ax_b.text(x0 - 0.035, ry + chh / 2, sym, fontsize=10, color=INK, ha='center', va='center')
    for ci, yr in enumerate(YEARS):
        hit = (sym == '♀' and yr == '1955')
        ax_b.add_patch(Rectangle((x0 + ci * cw, ry), cw * 0.88, chh,
                       facecolor=BLUE if hit else 'white',
                       edgecolor=BLUE if hit else GREY, lw=1.2 if hit else 0.7))
for ci, yr in enumerate(YEARS):
    ax_b.text(x0 + ci * cw + cw * 0.44, 0.40, yr, fontsize=5.4, color=MUT, ha='center', va='top')
ax_b.text(x0 + 1.5 * cw, 0.86, 'exact block:\nsame sex & birth year', fontsize=6.4,
          color=INK, ha='center', va='center', linespacing=1.25)
# arrow to caliper axis
ax_b.add_patch(FancyArrowPatch((0.40, 0.55), (0.48, 0.55), arrowstyle='-|>',
               mutation_scale=10, color=INK, lw=1.0))
# enrollment caliper axis
AXL, AXR, AXY = 0.52, 0.96, 0.5
ax_b.plot([AXL, AXR], [AXY, AXY], color=INK, lw=0.9)
cxl, cxr = 0.66, 0.82
ax_b.add_patch(Rectangle((cxl, AXY - 0.13), cxr - cxl, 0.26, facecolor=BLUE, alpha=0.12,
               edgecolor='none'))
for xe in (cxl, cxr):
    ax_b.plot([xe, xe], [AXY - 0.13, AXY + 0.13], color=BLUE, lw=0.7, ls=(0, (2, 2)))
ax_b.text((cxl + cxr) / 2, AXY + 0.15, 'enrollment\ncaliper', fontsize=6.2, color=BLUE,
          ha='center', va='bottom', linespacing=1.2)
ax_b.scatter(0.74, AXY, s=70, marker='D', color=BLUE, edgecolors='white', lw=0.7, zorder=5)
for cx in (0.70, 0.72, 0.77, 0.79):
    ax_b.scatter(cx, AXY, s=26, color=ORANGE, edgecolors='white', lw=0.5, zorder=4)
ax_b.text((AXL + AXR) / 2, 0.24, 'nearest controls by\nenrollment duration', fontsize=6.2,
          color=MUT, ha='center', va='top', linespacing=1.2)

# c: cheaters by design (table)
panel(ax_c, 'c'); ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis('off')
ax_c.text(0.0, 1.02, 'Shortcut predictors, fracture-timing AUROC', fontsize=8.2,
          fontweight='bold', color=INK, ha='left', va='top')
cCX = {'cut at\nfracture': 0.62, 'landmark': 0.90}
y = 0.80
for lab, cx in cCX.items():
    ax_c.text(cx, y, lab, fontsize=7.0, fontweight='bold', color=INK, ha='center',
              va='center', linespacing=1.2)
ax_c.plot([0, 1], [y - 0.11, y - 0.11], color=INK, lw=1.0)
y -= 0.24
for name in ('age gap', 'record shape'):
    cutv, lmv = c_data[name]
    ax_c.text(0.0, y, name, fontsize=7.2, color=INK, ha='left', va='center')
    ax_c.text(cCX['cut at\nfracture'], y, f'{cutv:.2f}', fontsize=7.2, color=ORANGE, ha='center', va='center')
    ax_c.text(cCX['landmark'], y, f'{lmv:.2f}', fontsize=7.2, color=BLUE, ha='center', va='center')
    y -= 0.20
ax_c.plot([0, 1], [y + 0.09, y + 0.09], color=INK, lw=1.0)
ax_c.text(0.0, y, 'Chance = 0.50. Predictors read only record structure, no\n'
          'diagnoses and no age. The age-gap channel collapses to chance\n'
          'under the landmark; record shape loses most of its signal.',
          fontsize=6.0, color=MUT, style='italic', ha='left', va='top', linespacing=1.45)

# d: fracture-timing discrimination by design (table)
panel(ax_d, 'd'); ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1); ax_d.axis('off')
ax_d.text(0.0, 1.02, 'Fracture-timing AUROC by cohort design', fontsize=8.2,
          fontweight='bold', color=INK, ha='left', va='top')
dCX = [('cut at\nfracture', ORANGE, 0.50), ('landmark', BLUE, 0.72), ('matched', GREEN, 0.92)]
y = 0.80
for lab, col, cx in dCX:
    ax_d.text(cx, y, lab, fontsize=7.0, fontweight='bold', color=col, ha='center',
              va='center', linespacing=1.2)
ax_d.plot([0, 1], [y - 0.11, y - 0.11], color=INK, lw=1.0)
y -= 0.22
d_cols = {'cut at\nfracture': [cut_frac[2], cut_frac[1], cut_frac[3]],
          'landmark': [lm['<1y'], lm['1-5y'], lm['5-10y']],
          'matched':  [mt['<1y'], mt['1-5y'], mt['5-10y']]}
for r, cls in enumerate(NAMES):
    ax_d.text(0.0, y, cls, fontsize=7.2, color=INK, ha='left', va='center')
    for lab, col, cx in dCX:
        ax_d.text(cx, y, f'{d_cols[lab][r]:.3f}', fontsize=7.2, color=INK, ha='center', va='center')
    y -= 0.15
ax_d.plot([0, 1], [y + 0.055, y + 0.055], color=INK, lw=0.5)
ax_d.text(0.0, y, 'Macro', fontsize=7.2, fontweight='bold', color=INK, ha='left', va='center')
for lab, col, cx in dCX:
    ax_d.text(cx, y, f'{np.mean(d_cols[lab]):.3f}', fontsize=7.2, fontweight='bold', color=INK, ha='center', va='center')
ax_d.plot([0, 1], [y - 0.09, y - 0.09], color=INK, lw=1.0)
ax_d.text(0.0, y - 0.13, 'Telling the fracture-timing classes apart, cases only.\n'
          'High under cut-at-fracture, then drops and settles once the\n'
          'window is fixed and the cohort is matched.',
          fontsize=6.0, color=MUT, style='italic', ha='left', va='top', linespacing=1.45)

for ext in ('png', 'pdf'):
    p = os.path.join(FIGS_DIR, f'fig_robustness.{ext}')
    fig.savefig(p, dpi=300 if ext == 'png' else None)
    print('wrote', p)
