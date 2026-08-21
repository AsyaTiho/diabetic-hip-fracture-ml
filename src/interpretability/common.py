"""Shared loading / scoring utilities for the BEHRT interpretability analyses.

Both interpretability scripts in this folder (run_occlusion.py and
run_attention_rollout.py) import from here so that the model is rebuilt and the
patient sequences are pre-processed *exactly* as in the evaluation pipeline
(src/external_validation/aou/eval_pipeline/inference/behrt_uchicago.py). Do not
diverge from that path -- any mismatch silently changes the predictions.

Run with the `ml` conda env:
    python run_occlusion.py
"""

import os
import gc
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Reuse the exact model + dataset definitions used at eval time ──────────
BEHRT_DIR = os.environ.get('HIPFX_BEHRT_CODE', str(Path(__file__).resolve().parents[1] / 'behrt'))
sys.path.insert(0, BEHRT_DIR)

from finetune_behrt_lora import (              # noqa: E402
    FinetuneDataset, BehrtForSequenceClassification,
    DATA_DIR, PRETRAIN_CKPT, FINETUNE_CKPT,
    MAX_LEN, NUM_LABELS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)
from MLM_model import BehrtConfig, BehrtForMaskedLM    # noqa: E402
from peft import LoraConfig, get_peft_model            # noqa: E402

UCHICAGO_DIR    = Path(os.environ.get('HIPFX_UCHICAGO_DIR', 'data/uchicago'))
UCHICAGO_OUTPUT = Path(DATA_DIR) / 'uchicago_output'

# Native class encoding (config.py CLASS_NAMES_NATIVE): 0 = control.
# Fracture risk (any horizon) = 1 - P(class 0), matching the paper's
# uc_risk_score. See the paper's machine-readable encoding note.
CONTROL_CLASS = 0

SPECIAL_TOKENS = {'CLS', 'SEP', 'PAD', 'MASK', 'UNK', '?'}

HERE     = Path(__file__).resolve().parent
OUT_DIR  = HERE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
#  Model
# ════════════════════════════════════════════════════════════════════════
def load_model(device, attn_eager=False):
    """Rebuild the LoRA-tuned BEHRT classifier exactly as behrt_uchicago.py.

    attn_eager=True forces the eager attention path so that
    output_attentions=True is honoured (needed for attention rollout).
    Returns (model, token2idx, age2idx, config).
    """
    with open(os.path.join(DATA_DIR, 'token2idx.pkl'), 'rb') as f:
        token2idx = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'age2idx.pkl'), 'rb') as f:
        age2idx = pickle.load(f)

    ckpt = torch.load(PRETRAIN_CKPT, map_location='cpu', weights_only=False)
    config = BehrtConfig(**ckpt['config_dict'])
    if attn_eager:
        # Force eager so the encoder returns attention probabilities.
        config._attn_implementation = 'eager'
        config.attn_implementation = 'eager'
    mlm_model = BehrtForMaskedLM(config)
    mlm_model.load_state_dict(ckpt['model_state_dict'])
    bert_backbone = mlm_model.bert
    del ckpt, mlm_model
    gc.collect()

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias='none',
    )
    get_peft_model(bert_backbone, lora_config)   # injects LoRA in-place

    model = BehrtForSequenceClassification(
        bert=bert_backbone,
        hidden_size=config.hidden_size,
        num_labels=NUM_LABELS,
        dropout=config.hidden_dropout_prob,
    )
    lora_ckpt = torch.load(FINETUNE_CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(lora_ckpt['model_state_dict'])
    del lora_ckpt
    gc.collect()

    model = model.to(device).eval()
    return model, token2idx, age2idx, config


# ════════════════════════════════════════════════════════════════════════
#  Data
# ════════════════════════════════════════════════════════════════════════
def load_uchicago_sequences(restrict_to_outcomes=True):
    """Return the UChicago sequence DataFrame (+ a sex_dict).

    Columns: enrolid, code (list[str]), age (list[str]), risk_class.
    If restrict_to_outcomes, also merges in the binary hip-fracture outcome
    (nt_fx_hip) and FRAX score so callers can pick cases for case studies.
    """
    seq = pd.read_pickle(UCHICAGO_OUTPUT / 'uchicago_dm_patient_sequences.pkl')
    with open(UCHICAGO_OUTPUT / 'uchicago_dm_sex_dict.pkl', 'rb') as f:
        sex_dict = pickle.load(f)

    if restrict_to_outcomes:
        outcome = pd.read_csv(UCHICAGO_DIR / 'Anony DM Outcome and FRAX.csv')
        outcome = outcome.rename(columns={'studyid': 'enrolid'})
        keep = [c for c in ('enrolid', 'nt_fx_hip', 'frax_hip_adj')
                if c in outcome.columns]
        seq = seq.merge(outcome[keep], on='enrolid', how='inner')
    return seq.reset_index(drop=True), sex_dict


# ════════════════════════════════════════════════════════════════════════
#  Scoring
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def score_dataframe(model, df, token2idx, age2idx, sex_dict, device,
                    batch_size=256):
    """Run the classifier on a DataFrame of sequences.

    df must have columns code, age, enrolid, risk_class (risk_class is unused
    for scoring but required by FinetuneDataset). Returns an (N, 4) softmax
    probability array in the order the rows appear in df.
    """
    from torch.utils.data import DataLoader
    ds = FinetuneDataset(df, token2idx, age2idx, MAX_LEN, sex_dict=sex_dict)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    probs = []
    for batch in loader:
        age, code, pos, seg, mask, sex, _ = [t.to(device) for t in batch]
        logits = model(code, age, seg, pos, sex, mask)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def fracture_risk(proba):
    """Any-horizon fracture risk = 1 - P(control).  proba: (N,4) -> (N,)."""
    return 1.0 - proba[:, CONTROL_CLASS]


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_pub_style():
    """Shared manuscript figure style: larger fonts, clean spines, editable PDF.

    Used by every figure in this folder so the panels are a consistent family.
    Keeps y-axis ticks (line plots need them); categorical bar charts hide their
    own y tick marks locally with ax.tick_params(axis='y', length=0).
    """
    import matplotlib
    matplotlib.rcParams.update({
        # Nature-style sans serif (Helvetica/Arial family), editable PDF text.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'TeX Gyre Heros',
                            'Arial', 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'stixsans',
        'font.size': 12,
        'axes.titlesize': 12, 'axes.labelsize': 13,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
        'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'axes.grid': False,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
