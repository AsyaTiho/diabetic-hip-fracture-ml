"""Diagnosis-level occlusion attribution for BEHRT-LoRA.

QUESTION ANSWERED
-----------------
"What if this patient had never been diagnosed with condition X?"
For each patient and each distinct diagnosis (phenotype token) in their record,
we delete *every* occurrence of that diagnosis from the sequence, re-score the
patient through the full model, and measure the change in predicted hip-fracture
risk (risk = 1 - P(control)). A large drop when a diagnosis is removed means the
model was leaning on that diagnosis to raise the patient's risk.

    delta_X = risk(full record) - risk(record without X)
    delta_X > 0  ->  diagnosis X was pushing risk UP
    delta_X < 0  ->  diagnosis X was pushing risk DOWN

Two deliverables:
  (1) GLOBAL importance (the XGBoost-feature-importance analog): aggregate
      delta_X over every patient who has diagnosis X -> mean signed effect,
      mean |effect|, and how often X is a patient's single biggest driver.
  (2) PER-PATIENT drivers: the top few diagnoses behind each patient's score,
      for clinician-facing "why was this patient flagged" explanations.

IMPORTANT CAVEAT (read before interpreting)
-------------------------------------------
Single-diagnosis occlusion measures each diagnosis's MARGINAL effect with the
rest of the record held fixed and still read in full context by the transformer.
It is therefore context-aware, but it is blind to REDUNDANCY and INTERACTIONS:
if two diagnoses carry overlapping signal, removing either alone barely moves the
score, so both look unimportant even though together they drive the prediction.
Cross-check the picture against run_attention_rollout.py, and read the README.

Usage (ml env):
    python run_occlusion.py                  # all patients with an outcome
    python run_occlusion.py --limit 500      # quick subset
    python run_occlusion.py --cases-only     # only hip-fracture cases (fast)
"""

import argparse
import time

import numpy as np
import pandas as pd

import common


def occlude_patient(model, row, token2idx, age2idx, sex_dict, device):
    """Return (baseline_risk, baseline_pred_class, list_of (token, delta, n_occ)).

    Builds one DataFrame holding the baseline sequence plus one variant per
    distinct non-special diagnosis (that diagnosis removed), scores them all in
    a single batch, and computes delta per diagnosis.
    """
    code = list(row['code'])
    age  = list(row['age'])

    # Distinct diagnoses to occlude, preserving first-seen order.
    seen = []
    seen_set = set()
    for c in code:
        if c in common.SPECIAL_TOKENS or c in seen_set:
            continue
        seen_set.add(c)
        seen.append(c)

    if not seen:
        return None

    rows = [row.to_dict()]            # row 0 = baseline (unchanged)
    kept = []                         # tokens that yield a non-empty variant
    for tok in seen:
        new_code, new_age = [], []
        for c, a in zip(code, age):
            if c == tok:
                continue
            new_code.append(c)
            new_age.append(a)
        # Skip if removing this diagnosis would empty the record (FinetuneDataset
        # indexes code[0] and crashes on a zero-length sequence). Such patients
        # have essentially a single diagnosis and no informative counterfactual.
        if len(new_code) == 0:
            continue
        d = row.to_dict()
        d['code'] = new_code
        d['age']  = new_age
        rows.append(d)
        kept.append(tok)
    seen = kept
    if not seen:
        return None

    df = pd.DataFrame(rows)
    proba = common.score_dataframe(model, df, token2idx, age2idx, sex_dict,
                                   device, batch_size=len(df))
    risk = common.fracture_risk(proba)

    base_risk = float(risk[0])
    base_pred = int(proba[0].argmax())
    results = []
    for i, tok in enumerate(seen, start=1):
        n_occ = sum(1 for c in code if c == tok)
        results.append((tok, base_risk - float(risk[i]), n_occ))
    return base_risk, base_pred, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='Score only the first N patients (0 = all).')
    ap.add_argument('--cases-only', action='store_true',
                    help='Restrict to hip-fracture cases (nt_fx_hip == 1).')
    ap.add_argument('--top-k', type=int, default=10,
                    help='Per-patient drivers to keep in the long table.')
    args = ap.parse_args()

    device = common.get_device()
    print(f'Device: {device}')
    print('Loading model...')
    model, token2idx, age2idx, _ = common.load_model(device)

    print('Loading UChicago sequences...')
    seq, sex_dict = common.load_uchicago_sequences(restrict_to_outcomes=True)

    if args.cases_only and 'nt_fx_hip' in seq.columns:
        seq = seq[seq['nt_fx_hip'] == 1].reset_index(drop=True)
        print(f'  cases-only: {len(seq):,} hip-fracture patients')
    if args.limit:
        seq = seq.iloc[:args.limit].reset_index(drop=True)
    print(f'  scoring {len(seq):,} patients')

    long_rows = []         # per (patient, diagnosis) effect
    patient_rows = []      # per-patient summary + top drivers
    t0 = time.time()
    for n, (_, row) in enumerate(seq.iterrows()):
        res = occlude_patient(model, row, token2idx, age2idx, sex_dict, device)
        if res is None:
            continue
        base_risk, base_pred, effects = res
        eid = int(row['enrolid'])

        effects_sorted = sorted(effects, key=lambda x: x[1], reverse=True)
        for tok, delta, n_occ in effects:
            long_rows.append({
                'enrolid': eid, 'diagnosis': tok, 'delta_risk': delta,
                'n_occurrences': n_occ, 'baseline_risk': base_risk,
            })

        top = effects_sorted[:args.top_k]
        prow = {
            'enrolid': eid,
            'baseline_risk': base_risk,
            'baseline_pred_class': base_pred,
            'n_distinct_diagnoses': len(effects),
        }
        if 'nt_fx_hip' in row:
            prow['nt_fx_hip'] = int(row['nt_fx_hip'])
        if 'risk_class' in row:
            prow['risk_class'] = int(row['risk_class'])
        for j, (tok, delta, _n) in enumerate(top, start=1):
            prow[f'driver{j}'] = tok
            prow[f'driver{j}_delta'] = round(delta, 5)
        patient_rows.append(prow)

        if (n + 1) % 200 == 0:
            dt = time.time() - t0
            rate = (n + 1) / dt
            print(f'  {n+1:,}/{len(seq):,}  ({rate:.1f} pt/s, '
                  f'eta {(len(seq)-n-1)/rate/60:.1f} min)')

    print(f'Done scoring in {(time.time()-t0)/60:.1f} min')

    long_df = pd.DataFrame(long_rows)
    patient_df = pd.DataFrame(patient_rows)

    # ── GLOBAL importance: the XGBoost-importance analog ──────────────────
    top_driver = (long_df.sort_values('delta_risk', ascending=False)
                          .groupby('enrolid').head(1))
    top_counts = top_driver['diagnosis'].value_counts()

    g = long_df.groupby('diagnosis')['delta_risk']
    glob = pd.DataFrame({
        'n_patients_with_dx': g.size(),
        'mean_delta_risk':    g.mean(),       # signed: + means raises risk
        'mean_abs_delta':     long_df.groupby('diagnosis')['delta_risk']
                                     .apply(lambda s: s.abs().mean()),
        'median_delta_risk':  g.median(),
    })
    glob['n_times_top_driver'] = top_counts
    glob['n_times_top_driver'] = glob['n_times_top_driver'].fillna(0).astype(int)
    glob = glob.sort_values('mean_abs_delta', ascending=False)

    long_path    = common.OUT_DIR / 'occlusion_long.csv'
    patient_path = common.OUT_DIR / 'occlusion_per_patient.csv'
    global_path  = common.OUT_DIR / 'occlusion_global_importance.csv'
    long_df.to_csv(long_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    glob.to_csv(global_path)
    print(f'\nWrote:\n  {long_path}\n  {patient_path}\n  {global_path}')

    print('\nTop 20 global drivers of predicted hip-fracture risk '
          '(by mean |delta|):')
    show = glob.head(20)[['n_patients_with_dx', 'mean_delta_risk',
                          'mean_abs_delta', 'n_times_top_driver']]
    with pd.option_context('display.width', 140,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(show.to_string())

    _make_figure(glob)


def _make_figure(glob, n=20):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover
        print(f'(skipping figure: {e})')
        return
    top = glob.sort_values('mean_delta_risk', ascending=False).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top) + 1))
    ax.barh(range(len(top)), top['mean_delta_risk'], color='#E41A1C')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel('Mean change in predicted hip-fracture risk when diagnosis '
                  'is removed\n(positive = diagnosis raises risk)')
    ax.set_title('BEHRT-LoRA occlusion importance (UChicago cohort)')
    ax.axvline(0, color='k', lw=0.6)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = common.OUT_DIR / f'occlusion_global_importance.{ext}'
        fig.savefig(p, dpi=150)
    print(f'Wrote figure -> {common.OUT_DIR}/occlusion_global_importance.png')


if __name__ == '__main__':
    main()
