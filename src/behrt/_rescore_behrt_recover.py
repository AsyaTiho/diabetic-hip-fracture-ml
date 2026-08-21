"""Recover enrolids for BEHRT's existing 50k predictions by reproducing the
deterministic eval split, validate order vs saved y_true, then re-score the
existing y_proba against the CANONICAL labels. NO GPU, NO inference.
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import label_binarize
def P(*a): print(*a, flush=True)

SEED=42; VAL_FRAC=0.15; EVAL_N=50_000
DIR=os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt')
S=['Control','<1y','1-5y','5-10y']

# 1. existing BEHRT predictions (native B-encoding 0=Ctrl,1=1-5y,2=<1y,3=5-10y)
z=np.load(f'{DIR}/behrt_4class_results_2026-05-22.npz',allow_pickle=True)
y_true_npz=z['y_true'].astype(int); proba_native=z['y_proba'].astype(np.float64)
P(f'npz: y_true={y_true_npz.shape} proba={proba_native.shape}')

# 2. reproduce eval_df deterministically
P('loading patient_sequences_finetune.pkl (22GB, keep enrolid+risk_class)...')
seq=pd.read_pickle(f'{DIR}/patient_sequences_finetune.pkl')
seq=seq[['enrolid','risk_class']].copy()
P(f'  rows={len(seq):,}')
train_df,val_df=train_test_split(seq,test_size=VAL_FRAC,random_state=SEED,stratify=seq['risk_class'])
val_df=val_df.reset_index(drop=True)
eval_df,_=train_test_split(val_df,train_size=EVAL_N,random_state=SEED,stratify=val_df['risk_class'])
eval_df=eval_df.reset_index(drop=True)
P(f'  reproduced eval_df: {len(eval_df):,}')

# 3. VALIDATE order: eval_df risk_class must equal npz y_true exactly
match=int((eval_df['risk_class'].values==y_true_npz).sum())
P(f'\nVALIDATION: eval_df.risk_class == npz.y_true : {match:,}/{len(eval_df):,}'
  f'  {"OK — enrolids aligned" if match==len(eval_df) else "MISMATCH — need GPU re-inference"}')
if match!=len(eval_df):
    P('Aborting re-score: order does not align; fall back to GPU inference.'); raise SystemExit

enrol=eval_df['enrolid'].astype(np.int64).values

# 4. canonical labels -> semantic ; B-encoding map {0:0,1:2,2:1,3:3}
canon=pd.read_pickle(f'{DIR}/hip_frac_labels_CANONICAL.pkl')
B2S={0:0,1:2,2:1,3:3}
canon_S=dict(zip(canon['enrolid'].astype(np.int64).values,
                 pd.Series(canon['risk_class']).map(B2S).astype(np.int64).values))
proba_S=proba_native[:,[0,2,1,3]]                       # -> [Ctrl,<1y,1-5y,5-10y]
orig_S=np.array([{0:0,2:1,1:2,3:3}[int(y)] for y in y_true_npz])
canon_y=np.array([canon_S.get(int(e),-1) for e in enrol])
keep=canon_y>=0
P(f'enrolids found in canonical: {int(keep.sum()):,}/{len(enrol):,}  (changed labels: {int((canon_y[keep]!=orig_S[keep]).sum())})')
proba_S=proba_S[keep]; orig_S=orig_S[keep]; canon_y=canon_y[keep]

def overall(y,pr):
    p=pr.argmax(1); yb=label_binarize(y,classes=[0,1,2,3])
    return (accuracy_score(y,p),precision_score(y,p,average='macro',zero_division=0),
            recall_score(y,p,average='macro',zero_division=0),f1_score(y,p,average='macro',zero_division=0),
            roc_auc_score(yb,pr,multi_class='ovr',average='macro'))
def diagnostic(y,pr):
    p=pr.argmax(1);o={}
    for k in range(4):
        bt=y==k;bp=p==k;TP=int((bt&bp).sum());FP=int((~bt&bp).sum());TN=int((~bt&~bp).sum());FN=int((bt&~bp).sum())
        o[k]=(TP/(TP+FN) if TP+FN else float('nan'),TP/(TP+FP) if TP+FP else float('nan'),TN/(TN+FN) if TN+FN else float('nan'))
    return o
def brier(y,pr):
    ci,ti=train_test_split(np.arange(len(y)),test_size=0.30,random_state=SEED,stratify=y)
    yf,prf=y[ci],pr[ci];yt,prt=y[ti],pr[ti];ybt=label_binarize(yt,classes=[0,1,2,3]);o={};iso=np.zeros_like(prt)
    for k in range(4):
        ir=IsotonicRegression(out_of_bounds='clip');ir.fit(prf[:,k],(yf==k).astype(float));iso[:,k]=ir.predict(prt[:,k])
    for k in range(4):
        t=ybt[:,k].astype(float);p=t.mean();o[k]=(p*(1-p),brier_score_loss(t,prt[:,k]),brier_score_loss(t,iso[:,k]))
    return o
def boot(y,pr,nb=1000):
    N=len(y);rng=np.random.default_rng(SEED);roc=np.full((nb,4),np.nan);apr=np.full((nb,4),np.nan)
    for b in range(nb):
        idx=rng.choice(N,N,replace=True);yb=y[idx];pb=pr[idx];yy=label_binarize(yb,classes=[0,1,2,3])
        for k in range(4):
            if yy[:,k].sum()==0:continue
            roc[b,k]=roc_auc_score(yy[:,k],pb[:,k]);apr[b,k]=average_precision_score(yy[:,k],pb[:,k])
    f=lambda a:(lambda x:f'{x.mean():.3f} ({np.percentile(x,2.5):.3f}--{np.percentile(x,97.5):.3f})')(a[~np.isnan(a)])
    return {k:(f(roc[:,k]),f(apr[:,k])) for k in range(4)}

P('\n=== BEHRT OVERALL (repro should match PAPER 0.968/0.412/0.821/0.496/0.973) ===')
P('repro: '+'  '.join(f'{v:.3f}' for v in overall(orig_S,proba_S)))
P('CANON: '+'  '.join(f'{v:.3f}' for v in overall(canon_y,proba_S)))
P('\n=== OPERATING POINT (repro | CANON) ===')
do=diagnostic(orig_S,proba_S);dc=diagnostic(canon_y,proba_S)
for mlabel,i in [('Sens',0),('PPV',1),('NPV',2)]:
    P(f'  {mlabel}: '+'  '.join(f'{S[k]}={do[k][i]:.3f}|{dc[k][i]:.3f}' for k in range(4)))
P('\n=== BRIER  base | raw_repro->CANON | iso_repro->CANON ===')
bo=brier(orig_S,proba_S);bc=brier(canon_y,proba_S)
for k in range(4):
    P(f'  {S[k]:7s} base {bc[k][0]:.4f} | raw {bo[k][1]:.4f}->{bc[k][1]:.4f} | iso {bo[k][2]:.4f}->{bc[k][2]:.4f}')
P('\n=== PER-CLASS ROC-AUC & AUPRC (bootstrap 1000) ===')
ro=boot(orig_S,proba_S);rc=boot(canon_y,proba_S)
for tag,r in [('repro',ro),('CANON',rc)]:
    P(f'  {tag}:')
    for k in range(4): P(f'    {S[k]:7s} ROC {r[k][0]:26s} AUPRC {r[k][1]}')
P('\nBEHRT DONE')
