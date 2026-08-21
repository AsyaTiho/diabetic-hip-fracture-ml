"""Discrimination among patients with NO prior bone-fragility diagnosis.

Answers the reviewer question: does BEHRT-LoRA still beat FRAX once we remove
the patients whose record already contains an obvious skeletal driver (any
fracture, osteoporosis, or multiple myeloma)? If yes, the model
is not merely re-flagging known bone disease; it extracts risk from the rest of
the record (diabetes comorbidity pattern, age, temporal structure).

This is a DISCRIMINATION test (AUROC/AUPRC), not an occlusion/attention analysis.
It reuses the saved predictions (enrolid-level) + the UChicago sequences (to find
who carries a bone code) + the outcome/FRAX CSV. No model run.

Sanity check: on the full FRAX subset it must reproduce the published
BEHRT 0.954 / FRAX 0.765 AUROC and 0.161 / 0.024 AUPRC. If it does, the
enrolid-level reconstruction is trustworthy and the subgroup numbers stand.

Run:  python run_subgroup_no_bone.py
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

import common

BEHRT_C = '#0072B2'   # Okabe-Ito blue
FRAX_C  = '#D55E00'   # Okabe-Ito vermillion

NPZ = Path(os.environ.get('HIPFX_RESULTS_DIR', 'results')) / 'outputs' / 'predictions' / 'uchicago_behrt.npz'
OUTCOME = Path(os.environ.get('HIPFX_UCHICAGO_DIR', 'data/uchicago')) / 'Anony DM Outcome and FRAX.csv'
SEQ = Path(os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')) / 'uchicago_output' / 'uchicago_dm_patient_sequences.pkl'

# "prior bone-fragility diagnosis": any fracture, osteoporosis, or myeloma
# token. (Hip-fracture phenotypes are already excluded from the
# sequences upstream, so they cannot appear here.)
BONE_RE = re.compile(r'fractur|osteoporos|myeloma', re.IGNORECASE)
SEED = 35
N_BOOT = 2000


def metrics(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


def boot_auc_delta(y, sb, sf, n_boot=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for BEHRT AUROC and the BEHRT-minus-FRAX AUROC delta."""
    rng = np.random.default_rng(seed)
    n = len(y)
    b_auc, d_auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n:
            continue
        ab = roc_auc_score(yi, sb[idx])
        af = roc_auc_score(yi, sf[idx])
        b_auc.append(ab)
        d_auc.append(ab - af)
    return (np.percentile(b_auc, [2.5, 97.5]),
            np.percentile(d_auc, [2.5, 97.5]))


def report(tag, df):
    y = df['y'].to_numpy()
    sb = df['risk'].to_numpy()
    sf = df['frax'].to_numpy()
    n, ev = len(df), int(y.sum())
    b_auc, b_ap = metrics(y, sb)
    f_auc, f_ap = metrics(y, sf)
    (blo, bhi), (dlo, dhi) = boot_auc_delta(y, sb, sf)
    print(f'\n=== {tag} ===')
    print(f'  N = {n:,}   hip fractures = {ev}  ({100*ev/n:.2f}%)')
    print(f'  AUROC   BEHRT {b_auc:.3f} [{blo:.3f}, {bhi:.3f}]   FRAX {f_auc:.3f}'
          f'   delta {b_auc-f_auc:+.3f} [{dlo:.3f}, {dhi:.3f}]')
    print(f'  AUPRC   BEHRT {b_ap:.3f}                   FRAX {f_ap:.3f}'
          f'   (prevalence {y.mean():.4f})')


def plot_subgroup(df):
    """ROC + PR for BEHRT vs FRAX in the no-bone subgroup, Nature style.

    Two separate PDFs (so the manuscript can place them as LaTeX subfigures,
    matching Figure 6). Okabe-Ito palette, no in-panel titles.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    common.set_pub_style()

    y = df['y'].to_numpy()
    sb, sf = df['risk'].to_numpy(), df['frax'].to_numpy()
    ab, _ = metrics(y, sb)
    af, _ = metrics(y, sf)
    apb = average_precision_score(y, sb)
    apf = average_precision_score(y, sf)
    prev = float(y.mean())

    # ROC
    fb, tb, _ = roc_curve(y, sb)
    ff, tf, _ = roc_curve(y, sf)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fb, tb, color=BEHRT_C, lw=2.4, label=f'BEHRT  (AUC = {ab:.3f})')
    ax.plot(ff, tf, color=FRAX_C,  lw=2.4, label=f'FRAX hip  (AUC = {af:.3f})')
    ax.plot([0, 1], [0, 1], ls=':', color='#999999', lw=1.2,
            label='Random classifier')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    # No in-panel title (the manuscript caption carries it, matching the other panels).
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    import matplotlib as _mpl
    with _mpl.rc_context({'savefig.bbox': 'standard'}):
        for ext in ('pdf', 'png'):  # fixed figure-size canvas -> ROC and PR identical
            fig.savefig(common.OUT_DIR / f'subgroup_no_bone_roc.{ext}')
    plt.close(fig)

    # PR
    pb, rb, _ = precision_recall_curve(y, sb)
    pf, rf, _ = precision_recall_curve(y, sf)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(rb, pb, color=BEHRT_C, lw=2.4, label=f'BEHRT  (AUPRC = {apb:.3f})')
    ax.plot(rf, pf, color=FRAX_C,  lw=2.4, label=f'FRAX hip  (AUPRC = {apf:.3f})')
    ax.axhline(prev, ls=':', color='#999999', lw=1.2,
               label=f'Baseline (prevalence = {prev:.3f})')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    # No in-panel title (the manuscript caption carries it, matching the other panels).
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    import matplotlib as _mpl
    with _mpl.rc_context({'savefig.bbox': 'standard'}):
        for ext in ('pdf', 'png'):  # fixed figure-size canvas -> ROC and PR identical
            fig.savefig(common.OUT_DIR / f'subgroup_no_bone_pr.{ext}')
    plt.close(fig)
    print(f'\nWrote subgroup_no_bone_roc/pr.{{pdf,png}} to {common.OUT_DIR}')


def main():
    z = np.load(NPZ, allow_pickle=True)
    risk_all = 1.0 - z['y_proba'][:, 0].astype(np.float64)   # 1 - P(control)
    pred = pd.DataFrame({'enrolid': z['enrolid'].astype(np.int64),
                         'risk': risk_all})

    out = pd.read_csv(OUTCOME).rename(columns={'studyid': 'enrolid'})
    out = out[['enrolid', 'nt_fx_hip', 'frax_hip_adj']].rename(
        columns={'nt_fx_hip': 'y', 'frax_hip_adj': 'frax'})

    seq = pd.read_pickle(SEQ)[['enrolid', 'code']]
    seq['has_bone'] = seq['code'].apply(
        lambda codes: any(BONE_RE.search(str(c)) for c in codes))

    df = (pred.merge(out, on='enrolid', how='inner')
              .merge(seq[['enrolid', 'has_bone']], on='enrolid', how='inner'))
    df = df[df['frax'].notna()].reset_index(drop=True)   # FRAX subset

    report('FULL FRAX subset (sanity: expect BEHRT 0.954/FRAX 0.765, '
           'AUPRC 0.161/0.024)', df)

    n_bone = int(df['has_bone'].sum())
    print(f'\nPatients with >=1 prior bone-fragility code: {n_bone:,} '
          f'({100*n_bone/len(df):.1f}%); '
          f'they carry {int(df.loc[df.has_bone, "y"].sum())} of '
          f'{int(df["y"].sum())} hip fractures.')

    report('NO prior bone-fragility diagnosis (key subgroup)',
           df[~df['has_bone']])
    report('WITH prior bone-fragility diagnosis (for contrast)',
           df[df['has_bone']])

    plot_subgroup(df[~df['has_bone']].reset_index(drop=True))


if __name__ == '__main__':
    main()
