"""Build hip_frac_labels_CANONICAL.pkl from the fresh 2026-06-08 cohort.

Re-port of the original _build_canonical_labels.py (which read the now-
deprecated df_with_dobyr.pkl).  This version reads phe_aggregated_2024.pkl
+ a small dobyr_lookup.pkl (both scp'd from mightymac).

Inputs (lambda, all in ~/embeddings_BEHRT/):
    phe_aggregated_2024.pkl    -- ~29 GB, scp'd from mightymac
    dobyr_lookup.pkl           -- ~150 MB, DataFrame[enrolid, dobyr],
                                  produced by extract_dobyr.py on mightymac

Output:
    data/behrt/hip_frac_labels_CANONICAL.pkl
        Columns: enrolid, diabetes_day, hip_frac_day, days_to_hip_frac, risk_class
        Rows: 12,033,245 (every patient in the cohort, controls included).

Risk-class encoding (CANONICAL; matches build_one_hot_features.py and the
BEHRT-LoRA fine-tune path):
    0 = Control      -- no hip fracture, or hip >10y after diabetes, or hip
                        before/same day as diabetes
    1 = 1-5 years    -- 365 < days_to_hip <= 1825
    2 = <1 year      -- 0 < days_to_hip <= 365
    3 = 5-10 years   -- 1825 < days_to_hip <= 3650

Encoding is byte-identical to the original _build_canonical_labels.py
rc() function.  Audit step removed (one-time historical reconciliation,
no longer needed).
"""

import os
import ast
import time

import numpy as np
import pandas as pd


DIR      = os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
AGG_PKL  = f'{DIR}/phe_aggregated_2024.pkl'
DOBYR    = f'{DIR}/dobyr_lookup.pkl'
OUT_PATH = f'{DIR}/hip_frac_labels_CANONICAL.pkl'

EXPLICIT = {'EM_202', 'EM_202.1', 'EM_202.2'}     # diabetes (BI_181 removed 2026-06-07)
HIPSET   = {'MS_745.11', 'MS_745.9'}              # hip fracture


def rc(dm, hf):
    """Risk class from (diabetes_day, hip_frac_day) Series.

    Returns int8 array, encoding: 0=Control, 1=1-5y, 2=<1y, 3=5-10y.
    """
    dt = hf - dm
    return np.where(
        dm.isna() | hf.isna(), 0,
        np.where(dt <= 0, 0,
        np.where(dt <= 365, 2,
        np.where(dt <= 1825, 1,
        np.where(dt <= 3650, 3, 0))))
    ).astype(np.int8)


def main():
    # ── 1. Load phe_aggregated_2024.pkl ──────────────────────────────────
    print(f'Loading {AGG_PKL} ...', flush=True)
    t0 = time.time()
    df = pd.read_pickle(AGG_PKL)
    print(f'  loaded {len(df):,} rows in {time.time() - t0:.1f}s', flush=True)
    print(f'  columns: {df.columns.tolist()}', flush=True)
    print(f'  unique patients: {df["enrolid"].nunique():,}', flush=True)

    # ── 2. Merge dobyr ───────────────────────────────────────────────────
    print(f'\nLoading {DOBYR} ...', flush=True)
    t0 = time.time()
    demo = pd.read_pickle(DOBYR)
    print(f'  {len(demo):,} rows in {time.time() - t0:.1f}s', flush=True)

    n_before = len(df)
    df = df.merge(demo, on='enrolid', how='inner')
    if n_before != len(df):
        print(f'  WARNING: lost {n_before - len(df):,} rows merging dobyr', flush=True)
    print(f'  after merge: {len(df):,} rows', flush=True)

    # ── 3. Mask diabetes + hip-fracture phecodes ─────────────────────────
    dia_mask = df['phecode'].isin(EXPLICIT)
    hip_mask = df['phecode'].isin(HIPSET)
    print(f'\nDiabetes phecode rows: {int(dia_mask.sum()):,}', flush=True)
    print(f'Hip-fracture phecode rows: {int(hip_mask.sum()):,}', flush=True)

    # ── 4. Explode all_intdays → one row per event; compute age ──────────
    def explode(mask, label):
        sub = df.loc[mask, ['enrolid', 'dobyr', 'phecode', 'all_intdays']].copy()

        def to_list(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return list(x)
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return []
            return [x]

        sub['days'] = sub['all_intdays'].map(to_list)
        sub = sub.explode('days').dropna(subset=['days'])
        sub['days'] = sub['days'].astype(np.int32)
        sub['age']  = (2003 + sub['days'] / 365.25 - sub['dobyr']).astype(int)
        n_total = len(sub)
        sub     = sub[sub['age'] >= 20]
        n_kept  = len(sub)
        print(f'  {label}: {n_total:,} events; '
              f'sub-20 dropped: {n_total - n_kept:,} '
              f'(should be 0 since upstream filter is active)', flush=True)
        return sub

    print('\nExploding event lists ...', flush=True)
    dia = explode(dia_mask, 'diabetes')
    hip = explode(hip_mask, 'hip fracture')

    # ── 5. First diabetes day + first hip-fracture day per patient ──────
    all_ids = pd.Index(df['enrolid'].drop_duplicates(), name='enrolid')
    diabetes_day = dia.groupby('enrolid')['days'].min()
    hip_frac_day = hip.groupby('enrolid')['days'].min()

    canon = pd.DataFrame(index=all_ids)
    canon['diabetes_day']     = diabetes_day
    canon['hip_frac_day']     = hip_frac_day
    canon['days_to_hip_frac'] = canon['hip_frac_day'] - canon['diabetes_day']
    canon['risk_class']       = rc(canon['diabetes_day'], canon['hip_frac_day'])

    # ── 6. Sanity print ─────────────────────────────────────────────────
    print('\nRisk-class distribution:', flush=True)
    name = {0: 'Control', 1: '1-5y', 2: '<1y', 3: '5-10y'}
    for c, n in canon['risk_class'].value_counts().sort_index().items():
        print(f'  {c} ({name[int(c)]}): {int(n):,}', flush=True)

    # ── 7. Save ─────────────────────────────────────────────────────────
    canon.reset_index().to_pickle(OUT_PATH)
    print(f'\nSaved {len(canon):,} rows to {OUT_PATH}', flush=True)
    print('DONE.', flush=True)


if __name__ == '__main__':
    main()
