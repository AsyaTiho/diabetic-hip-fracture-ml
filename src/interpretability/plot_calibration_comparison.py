"""Overlay raw vs recalibrated decision curves on one plot.

Visualises the calibration point from run_decision_curve.py: BEHRT's RAW score
(1 - P(control)) is a ranking score, so used directly its net benefit goes
negative and it loses to FRAX; after cross-validated recalibration the same model
dominates. Plotting both BEHRT curves together makes the "ranking tool, must be
recalibrated" message visible in a single figure.

Curves shown:
  - BEHRT (raw)          dashed red    -- uncalibrated, dips below zero
  - BEHRT (recalibrated) solid red     -- isotonic (default) or platt
  - FRAX (recalibrated)  solid blue
  - Screen all           grey dashed
  - Screen none          black (NB = 0)

Usage (ml env):
    python plot_calibration_comparison.py
    python plot_calibration_comparison.py --calib platt --tmax 0.10
"""

import argparse

import numpy as np

import common
from run_decision_curve import (net_benefit, treat_all_nb, recalibrate,
                                 NPZ, BOOTSTRAP_SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tmax', type=float, default=0.10)
    ap.add_argument('--tstep', type=float, default=0.001)
    ap.add_argument('--calib', choices=['isotonic', 'platt'], default='isotonic')
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    behrt_raw = d['uc_risk_score'].astype(np.float64)
    frax_raw  = d['frax_hip'].astype(np.float64) / 100.0
    y = d['y_hip'].astype(np.int64)
    N, n_events = len(y), int(y.sum())

    behrt_cal = recalibrate(behrt_raw, y, args.calib, 5, BOOTSTRAP_SEED)
    frax_cal  = recalibrate(frax_raw,  y, args.calib, 5, BOOTSTRAP_SEED)

    thr = np.arange(args.tstep, args.tmax + args.tstep / 2, args.tstep)
    nb_behrt_raw = net_benefit(behrt_raw, y, thr)
    nb_behrt_cal = net_benefit(behrt_cal, y, thr)
    nb_frax_cal  = net_benefit(frax_cal,  y, thr)
    nb_all = treat_all_nb(y, thr)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as e:
        print(f'matplotlib unavailable: {e}')
        return

    common.set_pub_style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(thr, nb_behrt_cal, color='#E41A1C', lw=2.2,
            label=f'BEHRT-LoRA (recalibrated, {args.calib})')
    ax.plot(thr, nb_behrt_raw, color='#E41A1C', lw=1.8, ls='--',
            label='BEHRT-LoRA (raw, uncalibrated)')
    ax.plot(thr, nb_frax_cal, color='#377EB8', lw=2.0,
            label=f'FRAX (recalibrated, {args.calib})')
    ax.plot(thr, nb_all, color='0.4', lw=1.2, ls='--', label='Screen all')
    ax.axhline(0, color='k', lw=1.0, label='Screen none')

    ax.set_xlim(thr.min(), thr.max())
    # Focus the y-axis on the region of interest; "Screen all" is allowed to run
    # off the bottom (standard DCA practice) so the BEHRT/FRAX curves are legible.
    ymax = max(nb_behrt_cal.max(), nb_frax_cal.max())
    ymin = min(nb_behrt_raw.min(), 0)
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel('Threshold probability (risk-score operating point)')
    ax.set_ylabel('Net benefit (net true positives per patient)')
    ax.set_title('Decision curve: effect of recalibration — UChicago cohort\n'
                 f'(N={N:,}, {n_events} hip fractures)')
    ax.legend(loc='lower left', frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = common.OUT_DIR / f'decision_curve_calibration_comparison.{ext}'
        fig.savefig(p, dpi=150)
    print(f'Wrote -> {common.OUT_DIR}/decision_curve_calibration_comparison.png')

    # quick console proof of the flip
    j = int(np.argmin(np.abs(thr - 0.02)))
    print(f'At threshold 0.02:  BEHRT raw NB={nb_behrt_raw[j]:+.5f}  '
          f'BEHRT calibrated NB={nb_behrt_cal[j]:+.5f}  '
          f'FRAX calibrated NB={nb_frax_cal[j]:+.5f}')


if __name__ == '__main__':
    main()
