"""Re-score the 3 MarketScan models against the CANONICAL labels
(explicit 4-code diabetes + age>=20). For each model: reproduce the
CURRENT (published) numbers using the labels stored in its results npz
(validation), then recompute using the canonical labels (NEW). Writes
everything to stdout (captured to a log) so it survives a window reload.

Semantic display order S = [Control, <1y, 1-5y, 5-10y].
 BEHRT native = B-encoding 0=Ctrl,1=1-5y,2=<1y,3=5-10y  -> reorder cols [0,2,1,3]; label map {0:0,2:1,1:2,3:3}
 XGB/NN native = D-encoding 0=Ctrl,1=<1y,2=1-5y,3=5-10y -> identity
Canonical risk_class is B-encoding -> S map {0:0,1:2,2:1,3:3}
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

SEED=42
S=['Control','<1y','1-5y','5-10y']
def P(*a): print(*a, flush=True)

canon=pd.read_pickle(os.path.join(os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt'), 'hip_frac_labels_CANONICAL.pkl'))
B2S={0:0,1:2,2:1,3:3}
canon_S=dict(zip(canon['enrolid'].astype(np.int64).values,
                 pd.Series(canon['risk_class']).map(B2S).astype(np.int64).values))
P(f'canonical labels loaded: {len(canon_S):,}')

MODELS={
 'BEHRT':  dict(f=os.path.join(os.environ.get('HIPFX_BEHRT_DIR', 'data/behrt'), 'behrt_4class_results_2026-05-22.npz'),
                reorder=[0,2,1,3], lab={0:0,2:1,1:2,3:3}, nboot=1000),
 'XGBoost':dict(f=os.path.join(os.environ.get('HIPFX_BASELINES_DIR', 'data/baselines'), 'xgboost_4class_results.npz'),
                reorder=[0,1,2,3], lab={0:0,1:1,2:2,3:3}, nboot=500),
 'NN':     dict(f=os.path.join(os.environ.get('HIPFX_BASELINES_DIR', 'data/baselines'), 'nn_4class_results.npz'),
                reorder=[0,1,2,3], lab={0:0,1:1,2:2,3:3}, nboot=500),
}
PAPER_OVERALL={'BEHRT':(0.968,0.412,0.821,0.496,0.973),
               'XGBoost':(0.686,0.258,0.474,0.222,0.764),
               'NN':(0.580,0.257,0.509,0.199,0.783)}

def load(m):
    d=np.load(m['f'],allow_pickle=True)
    proba=d['y_proba'][:,m['reorder']].astype(np.float64)
    orig=np.array([m['lab'][int(y)] for y in d['y_true']],dtype=np.int64)
    if 'enrolid' in d.files:
        e=d['enrolid'].astype(np.int64)
        canon=np.array([canon_S.get(int(x),-1) for x in e],dtype=np.int64)
        keep=canon>=0; miss=int((~keep).sum())
        return e[keep],proba[keep],orig[keep],canon[keep],miss
    else:
        return None,proba,orig,None,'NO ENROLID — canonical re-score pending'

def overall(y,proba):
    pred=proba.argmax(1)
    yb=label_binarize(y,classes=[0,1,2,3])
    return (accuracy_score(y,pred),
            precision_score(y,pred,average='macro',zero_division=0),
            recall_score(y,pred,average='macro',zero_division=0),
            f1_score(y,pred,average='macro',zero_division=0),
            roc_auc_score(yb,proba,multi_class='ovr',average='macro'))

def diagnostic(y,proba):
    pred=proba.argmax(1); out={}
    for k in range(4):
        bt=(y==k); bp=(pred==k)
        TP=int((bt&bp).sum());FP=int((~bt&bp).sum());TN=int((~bt&~bp).sum());FN=int((bt&~bp).sum())
        sens=TP/(TP+FN) if TP+FN else float('nan')
        ppv=TP/(TP+FP) if TP+FP else float('nan')
        npv=TN/(TN+FN) if TN+FN else float('nan')
        out[k]=(sens,ppv,npv)
    return out

def brier(y,proba):
    ci,ti=train_test_split(np.arange(len(y)),test_size=0.30,random_state=SEED,stratify=y)
    yf,prf=y[ci],proba[ci]; yt,prt=y[ti],proba[ti]
    ybt=label_binarize(yt,classes=[0,1,2,3]); out={}
    iso=np.zeros_like(prt)
    for k in range(4):
        ir=IsotonicRegression(out_of_bounds='clip'); ir.fit(prf[:,k],(yf==k).astype(float))
        iso[:,k]=ir.predict(prt[:,k])
    for k in range(4):
        t=ybt[:,k].astype(float); p=t.mean()
        out[k]=(p*(1-p),brier_score_loss(t,prt[:,k]),brier_score_loss(t,iso[:,k]))
    return out

def perclass_boot(y,proba,nboot):
    N=len(y); rng=np.random.default_rng(SEED)
    roc=np.full((nboot,4),np.nan); apr=np.full((nboot,4),np.nan)
    for b in range(nboot):
        idx=rng.choice(N,N,replace=True); yb=y[idx]; pb=proba[idx]
        yy=label_binarize(yb,classes=[0,1,2,3])
        for k in range(4):
            if yy[:,k].sum()==0: continue
            roc[b,k]=roc_auc_score(yy[:,k],pb[:,k]); apr[b,k]=average_precision_score(yy[:,k],pb[:,k])
    def fmt(a): a=a[~np.isnan(a)]; return f'{a.mean():.3f} ({np.percentile(a,2.5):.3f}--{np.percentile(a,97.5):.3f})'
    return {k:(fmt(roc[:,k]),fmt(apr[:,k])) for k in range(4)}

DATA={}
for name,m in MODELS.items():
    e,proba,orig,can,miss=load(m)
    DATA[name]=(e,proba,orig,can)
    P(f'\n{name}: N={len(proba):,}  canon={"YES" if can is not None else "NO ("+str(miss)+")"}')

P('\n'+'='*70); P('TABLE 1 — OVERALL (Acc, MacroP, MacroR, MacroF1, MacroAUC)'); P('='*70)
P(f'{"model":8s} {"which":7s}  Acc    P      R      F1     AUC')
for name in MODELS:
    e,proba,orig,can=DATA[name]
    o=overall(orig,proba); pp=PAPER_OVERALL[name]
    P(f'{name:8s} PAPER    '+'  '.join(f'{v:.3f}' for v in pp))
    P(f'{name:8s} repro    '+'  '.join(f'{v:.3f}' for v in o)+'   <- should match PAPER')
    if can is not None:
        c=overall(can,proba); P(f'{name:8s} CANON    '+'  '.join(f'{v:.3f}' for v in c)+'   <- NEW')
    else:
        P(f'{name:8s} CANON    pending enrolid')

P('\n'+'='*70); P('TABLE 3 — OPERATING POINT (Sensitivity / PPV / NPV)'); P('='*70)
for name in MODELS:
    e,proba,orig,can=DATA[name]
    do=diagnostic(orig,proba); dc=diagnostic(can,proba) if can is not None else None
    P(f'\n-- {name} --   '+('(repro | CANON)' if dc else '(repro; CANON pending enrolid)'))
    for metric,i in [('Sens',0),('PPV',1),('NPV',2)]:
        if dc: P(f'  {metric}: '+'  '.join(f'{S[k]}={do[k][i]:.3f}|{dc[k][i]:.3f}' for k in range(4)))
        else:  P(f'  {metric}: '+'  '.join(f'{S[k]}={do[k][i]:.3f}' for k in range(4)))

P('\n'+'='*70); P('TABLE 4 — BRIER (baseline / raw / isotonic)'); P('='*70)
for name in MODELS:
    e,proba,orig,can=DATA[name]
    bo=brier(orig,proba); bc=brier(can,proba) if can is not None else None
    P(f'\n-- {name} --')
    for k in range(4):
        if bc: P(f'  {S[k]:7s} base {bc[k][0]:.4f} | raw {bo[k][1]:.4f}->{bc[k][1]:.4f} | iso {bo[k][2]:.4f}->{bc[k][2]:.4f}')
        else:  P(f'  {S[k]:7s} base {bo[k][0]:.4f} | raw {bo[k][1]:.4f} | iso {bo[k][2]:.4f}   (repro; CANON pending)')

P('\n'+'='*70); P('TABLE 2 — PER-CLASS ROC-AUC & AUPRC (bootstrap CIs) [SLOW]'); P('='*70)
for name in MODELS:
    e,proba,orig,can=DATA[name]
    P(f'\n-- {name} (nboot={MODELS[name]["nboot"]}) --')
    P('  repro (orig labels):')
    ro=perclass_boot(orig,proba,MODELS[name]['nboot'])
    for k in range(4): P(f'    {S[k]:7s} ROC {ro[k][0]:26s} AUPRC {ro[k][1]}')
    if can is not None:
        P('  CANON (corrected labels):')
        rc=perclass_boot(can,proba,MODELS[name]['nboot'])
        for k in range(4): P(f'    {S[k]:7s} ROC {rc[k][0]:26s} AUPRC {rc[k][1]}')
    else:
        P('  CANON: pending enrolid')

P('\n\nALL DONE')
