"""Build UChicago external-validation figures (BEHRT-LoRA vs FRAX).

Reads outputs/predictions/uchicago_behrt.npz (produced by inference/behrt_uchicago.py).
Falls back to the stale 2026-05-22 npz if the new file doesn't exist.

Outputs in outputs/figs/:
  figUC_A_fracture_rate_decile.{pdf,png}    -- Observed hip fracture rate by risk decile
  figUC_B_roc_behrt_frax.{pdf,png}          -- ROC curves
  figUC_C_pr_behrt_frax.{pdf,png}           -- PR curves

Ported from publication_figures_2026-06-08.ipynb cells 29, 31, 35, 37.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UCHICAGO_PRED_OUT, UCHICAGO_PRED_FALLBACK, FIGS_DIR, BEHRT_COLOR, FRAX_COLOR


def _set_style():
    # One shared style for every data plot in the paper. See evaluation/figstyle.py.
    sns.set_style('ticks')
    import os as _o, sys as _s
    _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))  # -> evaluation/
    import figstyle
    figstyle.apply()


def _load():
    path = UCHICAGO_PRED_OUT if os.path.exists(UCHICAGO_PRED_OUT) else UCHICAGO_PRED_FALLBACK
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No UChicago predictions at either {UCHICAGO_PRED_OUT} or {UCHICAGO_PRED_FALLBACK}'
        )
    print(f'Loading UChicago predictions: {path}', flush=True)
    z = np.load(path, allow_pickle=True)
    return (z['uc_risk_score'].astype(np.float64),
            z['frax_hip'].astype(np.float64),
            z['y_hip'].astype(np.int64),
            path)


def _fracture_rates_by_decile(scores, y, q=10):
    try:
        dec   = pd.qcut(pd.Series(scores), q=q, labels=False, duplicates='drop')
        n_dec = int(dec.max()) + 1
    except (ValueError, TypeError):
        dec   = pd.Series(np.zeros(len(scores), dtype=int))
        n_dec = 1
    rates, ses, ns = [], [], []
    for d in range(n_dec):
        sel = (dec == d).to_numpy()
        m = int(sel.sum())
        if m == 0:
            rates.append(np.nan); ses.append(0.0); ns.append(0)
            continue
        p = float(np.mean(y[sel]))
        rates.append(p)
        ses.append(float(np.sqrt(p * (1 - p) / m)))
        ns.append(m)
    return np.array(rates), np.array(ses), np.array(ns), n_dec


def figUC_A(uc_risk, frax_hip, y_hip):
    rate_b, se_b, n_b, nd_b = _fracture_rates_by_decile(uc_risk,  y_hip)
    rate_f, se_f, n_f, nd_f = _fracture_rates_by_decile(frax_hip, y_hip)
    xb = np.arange(nd_b); xf = np.arange(nd_f)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(xb, rate_b * 100, yerr=se_b * 100,
                marker='o', capsize=3, linewidth=2.0, markersize=5,
                color=BEHRT_COLOR, label='BEHRT  (1 − P(>10 years))', zorder=3)
    ax.errorbar(xf, rate_f * 100, yerr=se_f * 100,
                marker='s', capsize=3, linewidth=2.0, markersize=5,
                color=FRAX_COLOR, label='FRAX hip %', zorder=2)
    n_ticks = max(nd_b, nd_f)
    ax.set_xticks(np.arange(n_ticks))
    ax.set_xticklabels([f'D{i+1}' for i in range(n_ticks)])
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    ax.set_xlabel('Risk decile  (D1 = lowest, D10 = highest)')
    ax.set_ylabel('Observed hip fracture rate (%)')
    ax.legend(loc='upper left', frameon=False)
    stem = os.path.join(FIGS_DIR, 'figUC_A_fracture_rate_decile')
    for ext in ('pdf', 'png'):
        plt.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  figUC_A: BEHRT D1={rate_b[0]*100:.2f}% D{nd_b}={rate_b[-1]*100:.2f}%  '
          f'FRAX D1={rate_f[0]*100:.2f}% D{nd_f}={rate_f[-1]*100:.2f}%', flush=True)


def figUC_B(uc_risk, frax_hip, y_hip):
    fpr_b, tpr_b, _ = roc_curve(y_hip, uc_risk);  auc_b = auc(fpr_b, tpr_b)
    fpr_f, tpr_f, _ = roc_curve(y_hip, frax_hip); auc_f = auc(fpr_f, tpr_f)
    label_b = f'BEHRT  (AUC = {auc_b:.3f})'
    label_f = f'FRAX hip  (AUC = {auc_f:.3f})'
    roc_df = pd.DataFrame(
        [{'FPR': f, 'TPR': t, 'Model': label_b} for f, t in zip(fpr_b, tpr_b)] +
        [{'FPR': f, 'TPR': t, 'Model': label_f} for f, t in zip(fpr_f, tpr_f)]
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.lineplot(data=roc_df, x='FPR', y='TPR', hue='Model',
                 hue_order=[label_b, label_f],
                 palette={label_b: BEHRT_COLOR, label_f: FRAX_COLOR},
                 linewidth=2.4,
                 estimator=None, sort=False, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle=':', color='#999999', linewidth=1.2,
            label='Random classifier')
    sns.despine(ax=ax)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', frameon=False)
    stem = os.path.join(FIGS_DIR, 'figUC_B_roc_behrt_frax')
    for ext in ('pdf', 'png'):
        plt.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  figUC_B: BEHRT AUC={auc_b:.3f}  FRAX AUC={auc_f:.3f}', flush=True)


def figUC_C(uc_risk, frax_hip, y_hip):
    prec_b, rec_b, _ = precision_recall_curve(y_hip, uc_risk);  ap_b = average_precision_score(y_hip, uc_risk)
    prec_f, rec_f, _ = precision_recall_curve(y_hip, frax_hip); ap_f = average_precision_score(y_hip, frax_hip)
    prevalence = float(y_hip.mean())
    label_b = f'BEHRT  (AUPRC = {ap_b:.3f})'
    label_f = f'FRAX hip  (AUPRC = {ap_f:.3f})'
    pr_df = pd.DataFrame(
        [{'Recall': r, 'Precision': p, 'Model': label_b} for r, p in zip(rec_b, prec_b)] +
        [{'Recall': r, 'Precision': p, 'Model': label_f} for r, p in zip(rec_f, prec_f)]
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.lineplot(data=pr_df, x='Recall', y='Precision', hue='Model',
                 hue_order=[label_b, label_f],
                 palette={label_b: BEHRT_COLOR, label_f: FRAX_COLOR},
                 linewidth=2.4,
                 estimator=None, sort=False, ax=ax)
    ax.axhline(prevalence, linestyle=':', color='#999999', linewidth=1.2,
               label=f'Baseline (prevalence = {prevalence:.3f})')
    sns.despine(ax=ax)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='upper right', frameon=False)
    stem = os.path.join(FIGS_DIR, 'figUC_C_pr_behrt_frax')
    for ext in ('pdf', 'png'):
        plt.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  figUC_C: BEHRT AUPRC={ap_b:.3f}  FRAX AUPRC={ap_f:.3f}  prev={prevalence:.3f}', flush=True)


def main():
    _set_style()
    uc_risk, frax_hip, y_hip, src = _load()
    print(f'  N = {len(y_hip):,}, hip fractures = {int(y_hip.sum())} ({100*y_hip.mean():.2f}%)', flush=True)
    print('\nMaking UChicago figures...', flush=True)
    figUC_A(uc_risk, frax_hip, y_hip)
    figUC_B(uc_risk, frax_hip, y_hip)
    figUC_C(uc_risk, frax_hip, y_hip)
    print(f'\nWrote 3 UChicago figures (source: {src}) to {FIGS_DIR}/', flush=True)


if __name__ == '__main__':
    main()
