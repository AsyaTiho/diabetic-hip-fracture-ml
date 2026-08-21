"""Shared paths, constants, and reuse policy for the unified eval pipeline."""

import os

PIPELINE_DIR = os.environ.get('HIPFX_RESULTS_DIR', 'results')
OUTPUTS_DIR  = os.path.join(PIPELINE_DIR, 'outputs')
PRED_DIR     = os.path.join(OUTPUTS_DIR, 'predictions')
FIGS_DIR     = os.path.join(OUTPUTS_DIR, 'figs')
TABLES_DIR   = os.path.join(OUTPUTS_DIR, 'tables')
LOGS_DIR     = os.path.join(PIPELINE_DIR, 'logs')

for d in (PRED_DIR, FIGS_DIR, TABLES_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ── Existing prediction sources (used when --skip-inference, the default) ──
# Dict ordering is LOAD-BEARING: BEHRT-LoRA MUST be last so the shared-rng
# bootstrap in tables/make_tables.py produces XGB/NN CIs byte-identical to
# the poster CSV in eval_archive_2026-06-15/poster_2026-06-08/.
MARKETSCAN_PRED_SOURCES = {
    'XGBoost':    os.path.join(os.environ.get('HIPFX_BASELINES_DIR', 'data/baselines'), 'xgboost_4class_results.npz'),
    'NN':         os.path.join(os.environ.get('HIPFX_BASELINES_DIR', 'data/baselines'), 'nn_4class_results.npz'),
    'BEHRT-LoRA': os.path.join(os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt'), 'behrt_4class_results_2026-06-08.npz'),
}

# Re-run outputs land here (one canonical place).
MARKETSCAN_PRED_OUT = {
    'XGBoost':    os.path.join(PRED_DIR, 'marketscan_xgb.npz'),
    'NN':         os.path.join(PRED_DIR, 'marketscan_nn.npz'),
    'BEHRT-LoRA': os.path.join(PRED_DIR, 'marketscan_behrt.npz'),
}

UCHICAGO_PRED_OUT = os.path.join(PRED_DIR, 'uchicago_behrt.npz')
# Fallback (stale, pre-BI_181-cleanup); used only if UChicago re-run is skipped.
UCHICAGO_PRED_FALLBACK = os.path.join(os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt'), 'uchicago_output', 'uchicago_frax_behrt_results_2026-05-22.npz')


def get_marketscan_pred(model_name, force_rerun=False):
    """Return path to the npz to read for `model_name`.
    Prefers a freshly re-run prediction in outputs/predictions/ when present;
    otherwise falls back to the canonical source npz.
    """
    rerun_path = MARKETSCAN_PRED_OUT[model_name]
    if not force_rerun and os.path.exists(rerun_path):
        return rerun_path
    if force_rerun:
        return rerun_path  # caller is expected to have populated it
    return MARKETSCAN_PRED_SOURCES[model_name]


# ── Encoding ────────────────────────────────────────────────────────────
# CANONICAL 4-class encoding (aligned across BEHRT, XGB, NN since 2026-06-08)
NUM_LABELS = 4
CLASS_NAMES_NATIVE = {
    0: '>10 year (control)',
    1: '1-5 years',
    2: '<1 year',
    3: '5-10 years',
}
# Figure display order = ascending duration to fracture
DISPLAY_NAMES   = ['<1 year', '1-5 years', '5-10 years', '>10 years']
DISPLAY_COLORS  = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Brewer Set1
# Map display slot d -> native class index in the npz
DISPLAY_TO_NATIVE = [2, 1, 3, 0]

# ── Bootstrap / calibration ─────────────────────────────────────────────
N_BOOTSTRAP   = 200
BOOTSTRAP_SEED = 35
ISO_CAL_SEED  = 42
ISO_CAL_TEST_FRAC = 0.30

# ── UChicago palette ────────────────────────────────────────────────────
BEHRT_COLOR = '#E41A1C'
FRAX_COLOR  = '#377EB8'

# ── Figure file stub per model ──────────────────────────────────────────
MODEL_STUB = {
    'BEHRT-LoRA': 'behrt',
    'XGBoost':    'xgboost',
    'NN':         'nn',
}
