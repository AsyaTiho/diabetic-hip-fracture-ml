"""Attention-rollout attribution for BEHRT-LoRA.

QUESTION ANSWERED
-----------------
"Which diagnoses in a patient's record does the model actually look at when it
forms its prediction?" Attention rollout (Abnar & Zuidema, 2020) traces how
information flows from each input token up through every transformer layer to
the [CLS] token that feeds the classifier. The rollout weight on a token is how
much of the [CLS] representation traces back to that token.

This is the CONTEXTUAL complement to occlusion: occlusion asks "what happens if
this diagnosis is gone"; attention rollout asks "what is the model reading."
Where the two agree, the explanation is robust. Where they disagree, attention
often reveals redundancy/interaction structure occlusion misses (see README).

Method
------
For each patient we run the encoder with output_attentions=True (eager attention
path forced in common.load_model), giving per-layer attention A_l of shape
[heads, L, L]. We:
  1. average over heads;
  2. add the identity (residual connection) and row-normalise;
  3. multiply the layer matrices together (rollout);
  4. read row 0 (the [CLS] query) -> attention mass landing on each token;
  5. zero out padding, renormalise, and aggregate by diagnosis name.

Deliverables mirror the occlusion script: a GLOBAL ranking (mean CLS-attention
per diagnosis across the cohort) and PER-PATIENT top attended diagnoses.

Usage (ml env):
    python run_attention_rollout.py
    python run_attention_rollout.py --limit 500
    python run_attention_rollout.py --cases-only
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import common
from finetune_behrt_lora import FinetuneDataset, MAX_LEN


class AttentionCatcher:
    """Capture per-layer attention probabilities via forward hooks.

    transformers v5 no longer threads ``output_attentions`` through the encoder,
    so we recompute the softmax(Q.K^T / sqrt(d) + mask) inside a hook on each
    self-attention module -- identical to the eager attention path, and
    independent of the library version. ``layers`` is filled in encoder order on
    every forward pass; call ``reset()`` before each batch.
    """

    def __init__(self, model):
        self.layers = []
        self.handles = []
        encoder = model.bert.encoder
        for layer in encoder.layer:
            sa = layer.attention.self
            self.handles.append(
                sa.register_forward_hook(self._hook, with_kwargs=True))

    def _hook(self, module, args, kwargs, output):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        attn_mask = kwargs.get('attention_mask', None)
        if attn_mask is None:                       # fall back: scan args
            for a in args[1:]:
                if torch.is_tensor(a) and a.dim() == 4:
                    attn_mask = a
                    break
        hd = module.attention_head_size
        B, L, _ = hidden.shape
        q = module.query(hidden).view(B, L, -1, hd).transpose(1, 2)
        k = module.key(hidden).view(B, L, -1, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask              # additive extended mask
        self.layers.append(torch.softmax(scores, dim=-1).detach())

    def reset(self):
        self.layers = []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def rollout_batch(model, catcher, batch, device):
    """Return (cls_attention [B,L], token_ids [B,L], mask [B,L]).

    cls_attention[b, j] = rollout weight from the CLS position onto position j.
    """
    age, code, pos, seg, mask, sex, _ = [t.to(device) for t in batch]

    catcher.reset()
    model.bert(code, age_ids=age, seg_ids=seg, posi_ids=pos, sex_ids=sex,
               attention_mask=mask)        # hooks fill catcher.layers in order

    rollout = None
    for A in catcher.layers:
        A = A.mean(dim=1)                # average heads -> [B,L,L]
        eye = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A = A + eye                      # residual
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = A if rollout is None else torch.bmm(A, rollout)

    cls_attn = rollout[:, 0, :]          # CLS query row -> [B,L]
    cls_attn = cls_attn * mask.to(cls_attn.dtype)   # drop padding
    cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return cls_attn.cpu().numpy(), code.cpu().numpy(), mask.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cases-only', action='store_true')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--top-k', type=int, default=10)
    args = ap.parse_args()

    device = common.get_device()
    print(f'Device: {device}')
    print('Loading model (eager attention)...')
    model, token2idx, age2idx, _ = common.load_model(device, attn_eager=True)
    idx2token = {v: k for k, v in token2idx.items()}
    catcher = AttentionCatcher(model)

    print('Loading UChicago sequences...')
    seq, sex_dict = common.load_uchicago_sequences(restrict_to_outcomes=True)
    if args.cases_only and 'nt_fx_hip' in seq.columns:
        seq = seq[seq['nt_fx_hip'] == 1].reset_index(drop=True)
        print(f'  cases-only: {len(seq):,} hip-fracture patients')
    if args.limit:
        seq = seq.iloc[:args.limit].reset_index(drop=True)
    print(f'  scoring {len(seq):,} patients')

    ds = FinetuneDataset(seq, token2idx, age2idx, MAX_LEN, sex_dict=sex_dict)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    enrolids = seq['enrolid'].values
    long_rows = []
    patient_rows = []
    ptr = 0
    t0 = time.time()
    for batch in loader:
        cls_attn, codes, masks = rollout_batch(model, catcher, batch, device)
        bsz = cls_attn.shape[0]
        for b in range(bsz):
            eid = int(enrolids[ptr + b])
            # Aggregate attention by diagnosis name (sum over repeats / visits).
            agg = {}
            for j in range(cls_attn.shape[1]):
                if masks[b, j] == 0:
                    continue
                name = idx2token.get(int(codes[b, j]), 'UNK')
                if name in common.SPECIAL_TOKENS:
                    continue
                agg[name] = agg.get(name, 0.0) + float(cls_attn[b, j])
            if not agg:
                continue
            total = sum(agg.values())
            ordered = sorted(agg.items(), key=lambda x: x[1], reverse=True)
            for name, w in ordered:
                long_rows.append({'enrolid': eid, 'diagnosis': name,
                                  'attention': w,
                                  'attention_frac': w / total})
            prow = {'enrolid': eid, 'n_distinct_diagnoses': len(agg)}
            row = seq.iloc[ptr + b]
            if 'nt_fx_hip' in seq.columns:
                prow['nt_fx_hip'] = int(row['nt_fx_hip'])
            if 'risk_class' in seq.columns:
                prow['risk_class'] = int(row['risk_class'])
            for k, (name, w) in enumerate(ordered[:args.top_k], start=1):
                prow[f'attended{k}'] = name
                prow[f'attended{k}_frac'] = round(w / total, 5)
            patient_rows.append(prow)
        ptr += bsz
        if ptr % (args.batch_size * 5) < args.batch_size:
            dt = time.time() - t0
            rate = ptr / dt
            print(f'  {ptr:,}/{len(seq):,}  ({rate:.0f} pt/s)')

    print(f'Done in {(time.time()-t0)/60:.1f} min')

    long_df = pd.DataFrame(long_rows)
    patient_df = pd.DataFrame(patient_rows)

    g = long_df.groupby('diagnosis')
    glob = pd.DataFrame({
        'n_patients_with_dx':  g.size(),
        'mean_attention_frac': g['attention_frac'].mean(),
        'sum_attention_frac':  g['attention_frac'].sum(),
    }).sort_values('mean_attention_frac', ascending=False)

    long_path    = common.OUT_DIR / 'attention_long.csv'
    patient_path = common.OUT_DIR / 'attention_per_patient.csv'
    global_path  = common.OUT_DIR / 'attention_global_importance.csv'
    long_df.to_csv(long_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    glob.to_csv(global_path)
    print(f'\nWrote:\n  {long_path}\n  {patient_path}\n  {global_path}')

    print('\nTop 20 most-attended diagnoses (mean attention fraction):')
    with pd.option_context('display.width', 140,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(glob.head(20).to_string())

    _make_figure(glob)


def _make_figure(glob, n=20):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover
        print(f'(skipping figure: {e})')
        return
    top = glob.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top) + 1))
    ax.barh(range(len(top)), top['mean_attention_frac'], color='#377EB8')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel('Mean CLS attention fraction (rollout)')
    ax.set_title('BEHRT-LoRA attention rollout (UChicago cohort)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(common.OUT_DIR / f'attention_global_importance.{ext}',
                    dpi=150)
    print(f'Wrote figure -> {common.OUT_DIR}/attention_global_importance.png')


if __name__ == '__main__':
    main()
