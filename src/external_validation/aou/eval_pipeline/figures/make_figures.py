"""Build all MarketScan figures (PR, ROC, CM) for the paper.

Outputs in outputs/figs/:
  fig1_pr_{behrt,xgboost,nn}.{pdf,png}
  fig2_roc_{behrt,xgboost,nn}.{pdf,png}
  fig4_cm_{behrt,xgboost,nn}.{pdf,png}

UChicago figs (figUC_A/B/C) are produced by figures/make_uchicago_figures.py
after behrt_uchicago.py runs.

Ported from publication_figures_2026-06-08.ipynb (now in
eval_archive_2026-06-15/old_scripts/) with identical plot params and seaborn
template so figures match the poster batch.
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')  # safe for headless runs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MARKETSCAN_PRED_SOURCES, MARKETSCAN_PRED_OUT, FIGS_DIR,
    NUM_LABELS, DISPLAY_NAMES, DISPLAY_COLORS, DISPLAY_TO_NATIVE,
    MODEL_STUB,
)


def _set_style():
    sns.set_style('ticks')
    sns.set_context('notebook')
    plt.rcParams['font.family']     = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Nimbus Sans', 'TeX Gyre Heros',
                                       'Liberation Sans', 'Arial', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'stixsans'


def _load_model(name, path):
    d = np.load(path, allow_pickle=True)
    probs  = d['y_proba'].astype(np.float64)
    labels = d['y_true'].astype(np.int64)
    y_bin  = label_binarize(labels, classes=list(range(NUM_LABELS)))
    print(f'  {name:<12}  probs={probs.shape}  N={len(labels):,}', flush=True)
    return {'probs': probs, 'labels': labels, 'y_bin': y_bin,
            'display_to_native': DISPLAY_TO_NATIVE}


def plot_pr(model_name, model, save_stem,
            log_y=False, log_x=False, linewidth=12.0, alpha=0.5):
    pr_rows, hue_order, palette, baselines = [], [], {}, []
    for color, display_idx, name in zip(DISPLAY_COLORS, range(NUM_LABELS), DISPLAY_NAMES):
        ni = model['display_to_native'][display_idx]
        prec_i, rec_i, _ = precision_recall_curve(model['y_bin'][:, ni],
                                                  model['probs'][:, ni])
        auprc_i      = average_precision_score(model['y_bin'][:, ni],
                                               model['probs'][:, ni])
        prevalence_i = model['y_bin'][:, ni].mean()
        label = f'{name} (AUPRC = {auprc_i:.3f})'
        hue_order.append(label); palette[label] = color
        baselines.append((color, prevalence_i))
        pr_rows.extend({'Recall': r, 'Precision': p, 'Class': label}
                       for r, p in zip(rec_i, prec_i))
    pr_df = pd.DataFrame(pr_rows)

    fig, ax = plt.subplots(figsize=(18, 17))
    sns.lineplot(data=pr_df, x='Recall', y='Precision', hue='Class',
                 hue_order=hue_order, palette=palette,
                 linewidth=linewidth, alpha=alpha,
                 estimator=None, sort=False, ax=ax)
    for color, prev in baselines:
        ax.axhline(prev, linestyle=':', color=color, linewidth=5.0, alpha=0.7)
    sns.despine(ax=ax)

    if log_x:
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1.05); ax.set_xticks([1e-4, 1e-2, 1.0])
    else:
        ax.set_xlim(0, 1.05); ax.set_xticks([0, 1.0])
    if log_y:
        ax.set_yscale('log'); ax.set_ylim(1e-4, 1.05); ax.set_yticks([1e-4, 1e-2, 1.0])
    else:
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 1.0])

    ax.tick_params(axis='both', labelsize=51)
    ax.set_xlabel('Recall',    fontsize=54, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=54, fontweight='bold')
    ax.set_title(f'{model_name} — PR-Curves for Each Class',
                 fontsize=60, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=True, facecolor='white', edgecolor='#cccccc',
              title=None, fontsize=45)
    for ext in ('pdf', 'png'):
        plt.savefig(f'{save_stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_roc(model_name, model, save_stem, linewidth=8.0, alpha=0.5):
    roc_rows, hue_order, palette = [], [], {}
    for color, display_idx, name in zip(DISPLAY_COLORS, range(NUM_LABELS), DISPLAY_NAMES):
        ni = model['display_to_native'][display_idx]
        fpr_i, tpr_i, _ = roc_curve(model['y_bin'][:, ni], model['probs'][:, ni])
        auc_i = auc(fpr_i, tpr_i)
        label = f'{name} (AUC = {auc_i:.3f})'
        hue_order.append(label); palette[label] = color
        roc_rows.extend({'FPR': f, 'TPR': t, 'Class': label}
                        for f, t in zip(fpr_i, tpr_i))
    roc_df = pd.DataFrame(roc_rows)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=roc_df, x='FPR', y='TPR', hue='Class',
                 hue_order=hue_order, palette=palette,
                 linewidth=linewidth, alpha=alpha,
                 estimator=None, sort=False, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle=':', color='#999999', linewidth=3.5)
    sns.despine(ax=ax)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis='both', labelsize=26)
    ax.set_xlabel('False Positive Rate', fontsize=32, fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=32, fontweight='bold')
    ax.set_title(f'{model_name} — ROC Curves for Each Class',
                 fontsize=32, fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, facecolor='white',
              edgecolor='#cccccc', title=None, fontsize=30)
    for ext in ('pdf', 'png'):
        plt.savefig(f'{save_stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cm(model_name, model, save_stem):
    d2n = model['display_to_native']
    y_true_display = np.zeros_like(model['labels'])
    y_pred_native  = model['probs'].argmax(axis=1)
    y_pred_display = np.zeros_like(y_pred_native)
    native_to_display = {d2n[d]: d for d in range(NUM_LABELS)}
    for ni, di in native_to_display.items():
        y_true_display[model['labels'] == ni] = di
        y_pred_display[y_pred_native == ni]   = di

    cm = confusion_matrix(y_true_display, y_pred_display,
                          labels=list(range(NUM_LABELS)))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues',
                   norm=LogNorm(vmin=1, vmax=cm.max()))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            val = cm[i, j]
            text = '0' if val == 0 else f'{val:,}'
            color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)
    ax.set_xticks(range(NUM_LABELS)); ax.set_yticks(range(NUM_LABELS))
    ax.set_xticklabels(DISPLAY_NAMES, fontsize=14)
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=14)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel('Predicted Class', fontsize=25, fontweight='bold')
    ax.set_ylabel('True Class',      fontsize=25, fontweight='bold')
    ax.set_title(f'{model_name} — Confusion Matrix',
                 fontsize=25, fontweight='bold', pad=20)
    sns.despine(ax=ax, left=True, bottom=True)
    for ext in ('pdf', 'png'):
        plt.savefig(f'{save_stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _resolve(force_rerun):
    paths = {}
    for name, src in MARKETSCAN_PRED_SOURCES.items():
        rerun = MARKETSCAN_PRED_OUT[name]
        paths[name] = rerun if (force_rerun or os.path.exists(rerun)) else src
    return paths


def main(force_rerun=False, models_subset=None):
    _set_style()
    print('Loading prediction npz files...', flush=True)
    paths = _resolve(force_rerun)
    if models_subset:
        paths = {k: v for k, v in paths.items() if k in models_subset}

    loaded = {n: _load_model(n, p) for n, p in paths.items()}

    print('\nMaking figures...', flush=True)
    for name, m in loaded.items():
        stub = MODEL_STUB[name]
        # XGB and NN use log_y=True per the poster recipe; BEHRT does not.
        log_y = (stub in ('xgboost', 'nn'))
        plot_pr (name, m, os.path.join(FIGS_DIR, f'fig1_pr_{stub}'),  log_y=log_y)
        plot_roc(name, m, os.path.join(FIGS_DIR, f'fig2_roc_{stub}'))
        plot_cm (name, m, os.path.join(FIGS_DIR, f'fig4_cm_{stub}'))
        print(f'  done: {stub}', flush=True)

    print(f'\nWrote figures to {FIGS_DIR}/', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-rerun', action='store_true')
    parser.add_argument('--models', default=None,
                        help='comma-separated subset: behrt,xgb,nn')
    args = parser.parse_args()
    subset = None
    if args.models:
        m = args.models.lower()
        subset = []
        if 'behrt' in m: subset.append('BEHRT-LoRA')
        if 'xgb'   in m: subset.append('XGBoost')
        if 'nn'    in m: subset.append('NN')
    main(force_rerun=args.force_rerun, models_subset=subset)
