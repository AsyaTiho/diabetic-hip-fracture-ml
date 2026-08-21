"""Regenerate the manuscript importance figures from the saved CSVs.

No model run needed -- reads outputs/occlusion_global_importance.csv and
outputs/attention_global_importance.csv. Produces, with a shared publication
style (larger fonts, thinned ticks):

  occlusion_importance_filtered.{png,pdf}   -- occlusion bar chart, n>=MIN_N
  attention_importance_filtered.{png,pdf}   -- attention bar chart, n>=MIN_N
  importance_tornado.{png,pdf}              -- back-to-back comparison

PREVALENCE FILTER (why): both rankings are conditional means over only the
patients who have a diagnosis, so a code seen in n=1 patient can top the chart on
noise alone. We keep only diagnoses present in >= MIN_N patients so the ranking
reflects drivers that matter across the population, not rare curiosities. The
filter is applied identically to both analyses; neither underlying computation
(occlusion Delta-risk, attention fraction) is changed.

Run:  python make_importance_figures.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import seaborn as sns

import common

MIN_N   = 50          # keep diagnoses present in >= MIN_N patients
TOP_N   = 15          # bars per single-analysis chart
TORN_K  = 10          # top-k from each analysis to union into the tornado
OCC_C   = '#D55E00'   # occlusion (Okabe-Ito vermillion)
ATT_C   = '#0072B2'   # attention (Okabe-Ito blue)


def _trunc(s, w=40):
    return s if len(s) <= w else s[:w - 1].rstrip() + '…'


def _label(name, n, w=40):
    return f'{_trunc(name, w)}  (n={int(n):,})'


def single_bar(df, value_col, fname, xlabel, title, color):
    """Horizontal importance bar chart, seaborn whitegrid styling.

    Labels are truncated short (SHORT_W chars) so the occlusion and attention
    panels stay legible when placed side by side in the manuscript figure.
    """
    SHORT_W = 22
    d = (df[df.n_patients_with_dx >= MIN_N]
         .sort_values(value_col, ascending=False).head(TOP_N)[::-1])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.2))
    y = range(len(d))
    ax.barh(list(y), d[value_col].values, color=color, height=0.74,
            edgecolor='white', linewidth=0.9, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([_label(n, c, SHORT_W) for n, c in
                        zip(d['diagnosis'], d['n_patients_with_dx'])])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis='x', color='#e3e3e3', lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True, top=True, right=True)
    ax.margins(y=0.01)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(xlabel)
    # No in-panel title: the manuscript caption carries it (Nature style).
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(common.OUT_DIR / f'{fname}.{ext}')
    plt.close(fig)
    print(f'wrote {fname}  ({len(d)} bars)')
    return d


def tornado(occ, att):
    o = occ[occ.n_patients_with_dx >= MIN_N].set_index('diagnosis')
    a = att[att.n_patients_with_dx >= MIN_N].set_index('diagnosis')
    order = list(o['mean_delta_risk'].sort_values(ascending=False).head(TORN_K).index)
    for d in a['mean_attention_frac'].sort_values(ascending=False).head(TORN_K).index:
        if d not in order:
            order.append(d)

    occ_v = o['mean_delta_risk'].reindex(order).clip(lower=0).fillna(0)
    att_v = a['mean_attention_frac'].reindex(order).fillna(0)
    n_pat = o['n_patients_with_dx'].reindex(order)
    # Normalise each side to its own maximum -> "relative importance" (the two
    # measures are different quantities, so only within-side magnitude is read).
    occ_n = occ_v / occ_v.max()
    att_n = att_v / att_v.max()

    order = order[::-1]                       # largest occlusion at top
    occ_n, att_n, n_pat = occ_n[order], att_n[order], n_pat[order]
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(order) + 1.6))
    ax.barh(list(y), -occ_n.values, color=OCC_C, height=0.74,
            label='Occlusion: changes the prediction')
    ax.barh(list(y),  att_n.values, color=ATT_C, height=0.74,
            label='Attention: what the model reads')
    ax.set_yticks(list(y))
    ax.set_yticklabels([_label(n, c) for n, c in zip(order, n_pat)])
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):.1f}'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel('Relative importance (each side scaled to its own maximum)')
    ax.text(-0.57, len(order) - 0.3, 'OCCLUSION', color=OCC_C,
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.57, len(order) - 0.3, 'ATTENTION', color=ATT_C,
            ha='center', va='center', fontsize=11, fontweight='bold')
    # No in-panel title: the manuscript caption carries it (Nature style).
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(common.OUT_DIR / f'importance_tornado.{ext}')
    plt.close(fig)
    print(f'wrote importance_tornado  ({len(order)} rows)')


def main():
    common.set_pub_style()
    occ = pd.read_csv(common.OUT_DIR / 'occlusion_global_importance.csv')
    att = pd.read_csv(common.OUT_DIR / 'attention_global_importance.csv')
    single_bar(occ, 'mean_delta_risk', 'occlusion_importance_filtered',
               'Mean drop in predicted risk',
               f'Occlusion importance (n≥{MIN_N} patients)', OCC_C)
    single_bar(att, 'mean_attention_frac', 'attention_importance_filtered',
               'Mean attention fraction (rollout)',
               f'Attention rollout (n≥{MIN_N} patients)', ATT_C)
    tornado(occ, att)


if __name__ == '__main__':
    main()
