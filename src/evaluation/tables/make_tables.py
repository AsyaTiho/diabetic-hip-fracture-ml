"""Build all MarketScan tables for the paper from prediction npz files.

Outputs (in outputs/tables/):
  table_overall_macro.{csv,tex}   -- Table 1: per-model macro Prec/Rec/F1/AUC
  table_per_class_ci.{csv,tex}    -- Table 3: per-class AUC + AUPRC with 95% bootstrap CI
  table_operating_point.{csv,tex} -- Table 6: per-class sens/PPV/NPV from argmax CM
  table_brier.{csv,tex}           -- Table 11: per-class Brier (raw + iso-calibrated)

Byte-identity rule:
  The original poster CI script (eval_archive_2026-06-15/poster_2026-06-08/figs_2026-06-08/
  compute_xgb_nn_ci.py) consumed the shared np.random.default_rng(35) state in this
  exact order per model: for c in [0,1,2,3]: AUC-bootstrap, AUPRC-bootstrap, iso-Brier.
  Reproducing that order means the XGBoost+NN columns of the new table_per_class_ci.csv
  must match the poster xgb_nn_ci_npv_brier_2026-06-09.csv byte-for-byte.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MARKETSCAN_PRED_SOURCES, MARKETSCAN_PRED_OUT, TABLES_DIR,
    NUM_LABELS, CLASS_NAMES_NATIVE,
    N_BOOTSTRAP, BOOTSTRAP_SEED, ISO_CAL_SEED,
)


def _resolve_pred_paths(force_rerun):
    """Return the {model: npz path} dict to read predictions from.
    Order matches MARKETSCAN_PRED_SOURCES (BEHRT-LoRA last) for byte-identity.
    """
    paths = {}
    for name, src in MARKETSCAN_PRED_SOURCES.items():
        rerun = MARKETSCAN_PRED_OUT[name]
        paths[name] = rerun if (force_rerun or os.path.exists(rerun)) else src
    return paths


def _bootstrap_ci(y_true, y_score, metric_fn, n_iter, rng):
    n = len(y_true)
    vals = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            vals[i] = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def _isotonic_brier(y_true_bin, y_score, rng):
    """70/30 split iso-calibration Brier — uses the shared rng for the permutation
    (matches compute_xgb_nn_ci.py byte-for-byte)."""
    n = len(y_true_bin)
    perm = rng.permutation(n)
    n_fit = int(0.7 * n)
    fit_idx, eval_idx = perm[:n_fit], perm[n_fit:]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_score[fit_idx], y_true_bin[fit_idx])
    p_cal = iso.transform(y_score[eval_idx])
    return float(brier_score_loss(y_true_bin[eval_idx], p_cal))


def _per_model_metrics(name, path, rng):
    """Compute every per-class number for one model. Returns list of dicts."""
    print(f'\n=== {name} ===', flush=True)
    print(f'  source: {path}', flush=True)
    t0 = time.time()

    z = np.load(path)
    y_true  = z['y_true'].astype(np.int64)
    y_pred  = z['y_pred'].astype(np.int64) if 'y_pred' in z.files else z['y_proba'].argmax(axis=1)
    y_proba = z['y_proba']
    n = len(y_true)
    print(f'  loaded {n:,} rows', flush=True)

    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_LABELS)), zero_division=0,
    )

    rows = []
    # Class loop — MUST be [0,1,2,3] (native order) and rng calls in
    # AUC→AUPRC→iso-Brier order, to preserve byte-identity with poster CSV.
    for c in range(NUM_LABELS):
        t1 = time.time()
        y_true_bin = (y_true == c).astype(np.int8)
        scores = y_proba[:, c]

        auc_mean, auc_lo, auc_hi = _bootstrap_ci(
            y_true_bin, scores, roc_auc_score, N_BOOTSTRAP, rng)
        ap_mean, ap_lo, ap_hi = _bootstrap_ci(
            y_true_bin, scores, average_precision_score, N_BOOTSTRAP, rng)

        # NPV from argmax confusion matrix (no rng)
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float('nan')

        brier_iso = _isotonic_brier(y_true_bin, scores, rng)
        brier_raw = float(brier_score_loss(y_true_bin, scores))

        rows.append({
            'model':      name,
            'class':      c,
            'class_name': CLASS_NAMES_NATIVE[c],
            'n_pos':      int(y_true_bin.sum()),
            'precision':  float(prec[c]),   # = PPV
            'recall':     float(rec[c]),    # = sensitivity
            'f1':         float(f1[c]),
            'auc_mean':   auc_mean,
            'auc_lo':     auc_lo,
            'auc_hi':     auc_hi,
            'auprc_mean': ap_mean,
            'auprc_lo':   ap_lo,
            'auprc_hi':   ap_hi,
            'npv':        npv,
            'brier_raw':  brier_raw,
            'brier_iso':  brier_iso,
        })
        print(f'  class {c} ({CLASS_NAMES_NATIVE[c]:<22s}): '
              f'AUC={auc_mean:.4f} ({auc_lo:.4f}-{auc_hi:.4f}), '
              f'AUPRC={ap_mean:.4f} ({ap_lo:.4f}-{ap_hi:.4f}), '
              f'NPV={npv:.4f}, Brier raw={brier_raw:.4f} iso={brier_iso:.4f} '
              f'[{time.time()-t1:.1f}s]', flush=True)

    # Macro / weighted aggregates (no rng — done after the bootstrap loop).
    y_bin = np.eye(NUM_LABELS, dtype=np.int8)[y_true]
    for avg in ('macro', 'weighted'):
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0,
        )
        try:
            auc_a = float(roc_auc_score(y_bin, y_proba, average=avg, multi_class='ovr'))
        except ValueError:
            auc_a = float('nan')
        rows.append({
            'model':      name,
            'class':      avg,
            'class_name': avg,
            'n_pos':      int(n),
            'precision':  float(prec_a),
            'recall':     float(rec_a),
            'f1':         float(f1_a),
            'auc_mean':   auc_a,
            'auc_lo':     float('nan'),
            'auc_hi':     float('nan'),
            'auprc_mean': float('nan'),
            'auprc_lo':   float('nan'),
            'auprc_hi':   float('nan'),
            'npv':        float('nan'),
            'brier_raw':  float('nan'),
            'brier_iso':  float('nan'),
        })

    print(f'  {name} done in {time.time()-t0:.1f}s', flush=True)
    return rows


def main(force_rerun=False):
    paths = _resolve_pred_paths(force_rerun)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    all_rows = []
    for name, path in paths.items():
        all_rows.extend(_per_model_metrics(name, path, rng))

    full = pd.DataFrame(all_rows)

    per_class = full[full['class'].apply(lambda x: isinstance(x, int))].reset_index(drop=True)
    agg = full[full['class'].isin(['macro', 'weighted'])].reset_index(drop=True)

    # ── Table 1: overall macro / weighted ────────────────────────────
    t1 = agg[['model', 'class', 'n_pos', 'precision', 'recall', 'f1', 'auc_mean']].rename(
        columns={'class': 'average', 'auc_mean': 'auc_ovr'}
    )
    t1.to_csv(os.path.join(TABLES_DIR, 'table_overall_macro.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'table_overall_macro.tex'), 'w') as f:
        f.write(t1.to_latex(index=False, float_format='%.3f', escape=False,
                            caption='Overall multi-class performance on the MarketScan held-out test set.',
                            label='tab:overall_macro'))

    # ── Table 3: per-class AUC + AUPRC with 95% bootstrap CI ─────────
    t3 = per_class[['model', 'class', 'class_name', 'n_pos',
                    'auc_mean', 'auc_lo', 'auc_hi',
                    'auprc_mean', 'auprc_lo', 'auprc_hi']]
    t3.to_csv(os.path.join(TABLES_DIR, 'table_per_class_ci.csv'),
              index=False)  # no float_format → preserves full precision for byte-diff
    with open(os.path.join(TABLES_DIR, 'table_per_class_ci.tex'), 'w') as f:
        f.write(t3.to_latex(index=False, float_format='%.3f', escape=False,
                            caption='Per-class discrimination on the MarketScan held-out test set with 95\\% bootstrap CIs.',
                            label='tab:per_class_ci'))

    # ── Table 6: per-class operating-point sens/PPV/NPV ──────────────
    t6 = per_class[['model', 'class', 'class_name', 'n_pos',
                    'recall', 'precision', 'npv']].rename(
        columns={'recall': 'sensitivity', 'precision': 'ppv'},
    )
    t6.to_csv(os.path.join(TABLES_DIR, 'table_operating_point.csv'), index=False)
    with open(os.path.join(TABLES_DIR, 'table_operating_point.tex'), 'w') as f:
        f.write(t6.to_latex(index=False, float_format='%.3f', escape=False,
                            caption='Operating-point classification performance on the MarketScan held-out test set, computed using one-vs-rest argmax predictions.',
                            label='tab:operating_point'))

    # ── Table 11: per-class Brier (raw + iso-calibrated) ─────────────
    t11 = per_class[['model', 'class', 'class_name', 'n_pos',
                     'brier_raw', 'brier_iso']]
    t11.to_csv(os.path.join(TABLES_DIR, 'table_brier.csv'),
               index=False)
    with open(os.path.join(TABLES_DIR, 'table_brier.tex'), 'w') as f:
        f.write(t11.to_latex(index=False, float_format='%.4f', escape=False,
                             caption='Per-class Brier scores before and after isotonic calibration on the MarketScan held-out test set.',
                             label='tab:brier'))

    print(f'\nWrote 4 tables to {TABLES_DIR}/', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-rerun', action='store_true',
                        help='read from outputs/predictions/ (assumes inference ran first)')
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
