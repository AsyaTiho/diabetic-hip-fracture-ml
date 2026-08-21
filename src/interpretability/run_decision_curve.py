"""Decision-curve / net-benefit analysis: BEHRT-LoRA vs FRAX (UChicago cohort).

QUESTION ANSWERED
-----------------
AUROC/AUPRC say how well a score *ranks* patients. A decision curve says whether
*acting on* the score is clinically worthwhile, and over what range of risk
thresholds. Net benefit puts true positives and false positives on one scale,
weighting false positives by the odds of the threshold probability p_t (how many
false alarms a clinician will tolerate to catch one true case):

    NB(p_t) = TP/N - (FP/N) * (p_t / (1 - p_t))

A patient is "screen-positive" at threshold p_t if their risk score >= p_t.
We compare four strategies across a grid of thresholds:
    - BEHRT-LoRA        (risk = 1 - P(control), the paper's uc_risk_score)
    - FRAX             (frax_hip 10-yr % / 100)
    - screen everyone  (treat-all)
    - screen no one    (treat-none, NB = 0)
The strategy with the highest net benefit at a given threshold is preferred
there. A screening tool should sit above treat-all and treat-none across the
low-threshold range where population pre-screening operates.

This needs no model: it reads the prediction arrays already written by
behrt_uchicago.py. Patient-level bootstrap (seed 35, matching the eval pipeline)
gives confidence bands and a BEHRT-minus-FRAX difference band -- important here
because the cohort has only 71 hip-fracture events.

CALIBRATION (important): decision-curve analysis assumes the score on the x-axis
is a *calibrated risk*. BEHRT's raw score (1 - P(control)) is a ranking score and
is badly miscalibrated (it runs up to ~1.0), so thresholding it as a probability
unfairly floods the screen-positive group with false positives. By default we
therefore recalibrate BOTH scores to the binary hip outcome using cross-validated
isotonic regression (5-fold, pooled out-of-fold predictions, seed 35) before
computing net benefit -- this is the standard, fair way to give a ranking tool a
DCA, and the cross-validation keeps it honest (no patient is calibrated on
itself). Use --raw for the uncalibrated curve or --calib platt for parametric
(logistic) recalibration.

Usage (ml env):
    python run_decision_curve.py                 # isotonic-recalibrated (default)
    python run_decision_curve.py --raw           # uncalibrated scores
    python run_decision_curve.py --calib platt --tmax 0.10
"""

import os
import argparse
from pathlib import Path

import numpy as np

import common


def recalibrate(score, y, method='isotonic', n_splits=5, seed=35):
    """Cross-validated recalibration -> pooled out-of-fold probabilities.

    Maps an arbitrary score onto the binary outcome y so the result is a
    calibrated risk suitable for net-benefit thresholding. Out-of-fold pooling
    avoids the optimism of calibrating and evaluating on the same patients.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    eps = 1e-6
    s = np.clip(score.astype(np.float64), eps, 1 - eps)
    oof = np.full(len(s), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(s, y):
        if method == 'isotonic':
            ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            ir.fit(s[tr], y[tr])
            oof[te] = ir.predict(s[te])
        else:  # platt: logistic regression on the logit of the score
            x_tr = np.log(s[tr] / (1 - s[tr])).reshape(-1, 1)
            x_te = np.log(s[te] / (1 - s[te])).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver='lbfgs')
            lr.fit(x_tr, y[tr])
            oof[te] = lr.predict_proba(x_te)[:, 1]
    return oof

NPZ = Path(os.environ.get('HIPFX_RESULTS_DIR', 'results')) / 'outputs' / 'predictions' / 'uchicago_behrt.npz'
BOOTSTRAP_SEED = 35          # matches eval_pipeline config.py
REPORT_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05]


def net_benefit(score, y, thresholds):
    """Vectorised net benefit over a 1-D array of thresholds.

    score, y : (N,) ; thresholds : (T,) -> returns (T,) net benefit.
    """
    N = len(y)
    n_pos = y.sum()
    nb = np.empty(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        pred = score >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb[i] = tp / N - (fp / N) * (t / (1.0 - t))
    return nb


def treat_all_nb(y, thresholds):
    N = len(y)
    prev = y.sum() / N
    return prev - (1.0 - prev) * (thresholds / (1.0 - thresholds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tmax', type=float, default=0.10,
                    help='Max threshold probability on the x-axis.')
    ap.add_argument('--tstep', type=float, default=0.001)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--raw', action='store_true',
                    help='Skip recalibration; use raw scores (uncalibrated).')
    ap.add_argument('--calib', choices=['platt', 'isotonic'], default='isotonic',
                    help='Recalibration method (default isotonic).')
    ap.add_argument('--n-splits', type=int, default=5)
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    behrt = d['uc_risk_score'].astype(np.float64)   # 1 - P(control), in [0,1]
    frax  = d['frax_hip'].astype(np.float64) / 100.0  # % -> probability
    y     = d['y_hip'].astype(np.int64)
    N = len(y)
    n_events = int(y.sum())
    print(f'UChicago FRAX subset: N={N:,}, hip fractures={n_events} '
          f'({100*n_events/N:.2f}%)')
    print(f'Raw BEHRT range [{behrt.min():.4f}, {behrt.max():.4f}]; '
          f'raw FRAX range [{frax.min():.4f}, {frax.max():.4f}]')

    if args.raw:
        calib_tag = 'raw (uncalibrated)'
    else:
        behrt = recalibrate(behrt, y, args.calib, args.n_splits, BOOTSTRAP_SEED)
        frax  = recalibrate(frax,  y, args.calib, args.n_splits, BOOTSTRAP_SEED)
        calib_tag = f'{args.calib} recalibrated ({args.n_splits}-fold CV)'
        print(f'Recalibrated both scores: {calib_tag}')
        print(f'Calib BEHRT range [{behrt.min():.4f}, {behrt.max():.4f}]; '
              f'calib FRAX range [{frax.min():.4f}, {frax.max():.4f}]')

    thr = np.arange(args.tstep, args.tmax + args.tstep / 2, args.tstep)

    nb_behrt = net_benefit(behrt, y, thr)
    nb_frax  = net_benefit(frax,  y, thr)
    nb_all   = treat_all_nb(y, thr)
    nb_none  = np.zeros_like(thr)

    # ── Patient-level bootstrap for CIs + BEHRT-minus-FRAX band ───────────
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bb = np.empty((args.n_boot, len(thr)))
    bf = np.empty((args.n_boot, len(thr)))
    for b in range(args.n_boot):
        idx = rng.integers(0, N, N)
        bb[b] = net_benefit(behrt[idx], y[idx], thr)
        bf[b] = net_benefit(frax[idx],  y[idx], thr)
    behrt_lo, behrt_hi = np.percentile(bb, [2.5, 97.5], axis=0)
    frax_lo,  frax_hi  = np.percentile(bf, [2.5, 97.5], axis=0)
    diff = bb - bf
    diff_mean = diff.mean(axis=0)
    diff_lo, diff_hi = np.percentile(diff, [2.5, 97.5], axis=0)

    # ── Save full curve ──────────────────────────────────────────────────
    import pandas as pd
    out = pd.DataFrame({
        'threshold': thr,
        'nb_behrt': nb_behrt, 'nb_behrt_lo': behrt_lo, 'nb_behrt_hi': behrt_hi,
        'nb_frax': nb_frax,   'nb_frax_lo': frax_lo,   'nb_frax_hi': frax_hi,
        'nb_treat_all': nb_all, 'nb_treat_none': nb_none,
        'nb_diff_behrt_minus_frax': diff_mean,
        'nb_diff_lo': diff_lo, 'nb_diff_hi': diff_hi,
    })
    suffix = 'raw' if args.raw else args.calib
    curve_path = common.OUT_DIR / f'decision_curve_{suffix}.csv'
    out.to_csv(curve_path, index=False)

    # ── Summary at clinically relevant thresholds ────────────────────────
    rows = []
    for t in REPORT_THRESHOLDS:
        j = int(np.argmin(np.abs(thr - t)))
        rows.append({
            'threshold': round(thr[j], 4),
            'nb_behrt': round(nb_behrt[j], 6),
            'nb_frax': round(nb_frax[j], 6),
            'nb_treat_all': round(nb_all[j], 6),
            'behrt_minus_frax': round(diff_mean[j], 6),
            'diff_95ci': f'[{diff_lo[j]:.6f}, {diff_hi[j]:.6f}]',
            # interpretive: net true positives per 1000 screened vs treat-none
            'behrt_net_TP_per_1000': round(nb_behrt[j] * 1000, 2),
        })
    summary = pd.DataFrame(rows)
    summary_path = common.OUT_DIR / f'decision_curve_summary_{suffix}.csv'
    summary.to_csv(summary_path, index=False)

    # Range where BEHRT dominates both FRAX and treat-all.
    dominates = (nb_behrt > nb_frax) & (nb_behrt > nb_all) & (nb_behrt > 0)
    dom_thr = thr[dominates]
    print(f'\nWrote:\n  {curve_path}\n  {summary_path}')
    print('\nNet benefit at clinically relevant thresholds:')
    with pd.option_context('display.width', 160):
        print(summary.to_string(index=False))
    if dom_thr.size:
        print(f'\nBEHRT dominates FRAX, treat-all and treat-none over '
              f'threshold {dom_thr.min():.3f}-{dom_thr.max():.3f}.')
    else:
        print('\nBEHRT does not strictly dominate across the grid '
              '(inspect the curve).')

    _make_figure(thr, nb_behrt, behrt_lo, behrt_hi, nb_frax, frax_lo, frax_hi,
                 nb_all, n_events, N, calib_tag, suffix)


def _make_figure(thr, nb_behrt, b_lo, b_hi, nb_frax, f_lo, f_hi, nb_all,
                 n_events, N, calib_tag, suffix):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as e:                       # pragma: no cover
        print(f'(skipping figure: {e})')
        return
    common.set_pub_style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.fill_between(thr, b_lo, b_hi, color='#0072B2', alpha=0.15, lw=0)
    ax.fill_between(thr, f_lo, f_hi, color='#D55E00', alpha=0.15, lw=0)
    ax.plot(thr, nb_behrt, color='#0072B2', lw=2, label='BEHRT-LoRA')
    ax.plot(thr, nb_frax,  color='#D55E00', lw=2, label='FRAX')
    ax.plot(thr, nb_all, color='0.4', lw=1.2, ls='--', label='Screen all')
    ax.axhline(0, color='k', lw=1.0, label='Screen none')
    ax.set_xlim(thr.min(), thr.max())
    # Focus the y-axis on the BEHRT/FRAX region of interest; "Screen all" is
    # allowed to run off the bottom (standard DCA practice) so the curves of
    # interest are legible instead of compressed against the top.
    ymax = max(nb_behrt.max(), nb_frax.max(), 0.0)
    ymin = min(nb_behrt.min(), nb_frax.min(), 0.0)
    pad = 0.18 * (ymax - ymin) + 1e-4
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel('Threshold probability (risk-score operating point)')
    ax.set_ylabel('Net benefit (net true positives per patient)')
    # No in-panel title: the manuscript caption carries it (Nature style).
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(common.OUT_DIR / f'decision_curve_{suffix}.{ext}', dpi=150)
    print(f'Wrote figure -> {common.OUT_DIR}/decision_curve_{suffix}.png')


if __name__ == '__main__':
    main()
