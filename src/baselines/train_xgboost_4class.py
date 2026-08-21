#!/usr/bin/env python
"""Train XGBoost for 4-class hip fracture risk prediction with Optuna HPO.

Classes (canonical encoding, matches _build_canonical_labels.py and the
BEHRT-LoRA fine-tune labels -- note classes 1 and 2 are swapped vs. the
natural chronological order, on purpose, so the two pipelines share the
same integer-to-meaning map):
  0 – Control (no hip fracture or > 10 yr)   ~99.3 %
  1 – Hip fracture 1–5 years                 ~0.48 %
  2 – Hip fracture within 1 year             ~0.31 %
  3 – Hip fracture 5–10 years                ~0.19 %

Multi-GPU strategy (XGBoost trains one model per GPU):
  Optuna runs N_JOBS=4 trials in parallel, each trial pinned to a different GPU
  via AVAILABLE_GPUS[trial.number % 4].  The final model trains on cuda:0 only
  (XGBoost hist does not support multi-GPU for a single model).

Pipeline:
  1. Downsample class 0 to CLASS0_CAP in training only; val/test keep full dist.
  2. Optuna HPO: 4 parallel trials, StratifiedKFold, QuantileDMatrix, MedianPruner.
  3. Final model: best params + avg best_iteration from CV, early stopping on val.

Outputs (all in same directory as this script):
  xgboost_4class_model.json
  optuna_study_xgboost_4class.pkl
  xgboost_4class_results.npz

Usage:
  cd Testing_NN_XGBoost_on_4_risk_classes
  python train_xgboost_4class.py
"""

import gc
import os
import time
from datetime import timedelta

import joblib
import numpy as np
import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from scipy.sparse import load_npz
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# ── paths ──────────────────────────────────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE   = os.path.join(DATA_DIR, 'one_hot_features_4groups.npz')
META_FILE   = os.path.join(DATA_DIR, 'one_hot_labels_metadata_4groups.npz')
MODEL_OUT   = os.path.join(DATA_DIR, 'xgboost_4class_model.json')
STUDY_OUT   = os.path.join(DATA_DIR, 'optuna_study_xgboost_4class.pkl')
RESULTS_OUT = os.path.join(DATA_DIR, 'xgboost_4class_results.npz')

# ── config ─────────────────────────────────────────────────────
SEED = 35

CLASS0_CAP = 600_000

VAL_FRAC  = 0.10
TEST_FRAC = 0.10

N_FOLDS  = 5
N_TRIALS = 60
N_JOBS   = 4   # parallel Optuna trials = number of GPUs

NUM_BOOST_ROUND       = 500
EARLY_STOPPING_ROUNDS = 40   # per CV fold
FINAL_EARLY_STOPPING  = 50   # final model training

AVAILABLE_GPUS = [0, 1, 2, 3]

# CPU threads per trial — capped so N_JOBS * NTHREAD ≤ cpu_count
_cpu_count = os.cpu_count() or 16
NTHREAD_PER_TRIAL = max(1, _cpu_count // N_JOBS)

GROUP_LABELS = [
    'Control (no hip frac or >10yr)',
    'Hip frac 1–5 years',         # class 1
    'Hip frac within 1 year',     # class 2 (canonical encoding swap)
    'Hip frac 5–10 years',        # class 3
]
NUM_CLASSES = 4


# ── helpers ────────────────────────────────────────────────────
def gpu_for_trial(trial_number: int) -> str:
    """Return cuda:N string, falling back to cpu if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return f'cuda:{AVAILABLE_GPUS[trial_number % len(AVAILABLE_GPUS)]}'
    except ImportError:
        pass
    return 'cpu'


def print_dist(y, label=''):
    unique, counts = np.unique(y, return_counts=True)
    print(f'  {label}')
    for cls, cnt in zip(unique, counts):
        print(f'    class {cls} ({GROUP_LABELS[cls]}): {cnt:,}  ({100*cnt/len(y):.2f}%)')


def downsample_class0(X, y, cap, rng):
    idx0     = np.where(y == 0)[0]
    idx_rest = np.where(y != 0)[0]
    if len(idx0) > cap:
        idx0 = rng.choice(idx0, size=cap, replace=False)
    keep = np.concatenate([idx0, idx_rest])
    rng.shuffle(keep)
    return X[keep], y[keep]


def eval_split(model, X, y_true, split_name):
    proba = model.predict(xgb.DMatrix(X))   # (N, 4)
    preds = proba.argmax(axis=1)
    ll  = log_loss(y_true, proba)
    wf1 = f1_score(y_true, preds, average='weighted', zero_division=0)
    mf1 = f1_score(y_true, preds, average='macro',    zero_division=0)
    try:
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    print(f'\n── {split_name} ──')
    print(f'  Log-loss   : {ll:.4f}')
    print(f'  Weighted F1: {wf1:.4f}')
    print(f'  Macro F1   : {mf1:.4f}')
    print(f'  Macro AUC  : {auc:.4f}')
    print(classification_report(y_true, preds, target_names=GROUP_LABELS, zero_division=0))
    print('  Confusion matrix:')
    print(confusion_matrix(y_true, preds))
    return proba, preds


# ── Optuna callback ────────────────────────────────────────────
class TimingCallback:
    """Thread-safe: reads timing from trial metadata, not from a shared attribute."""

    def __init__(self):
        self._best_value   = None
        self._trials_since = 0
        self._lock         = __import__('threading').Lock()

    def __call__(self, study, trial):
        if trial.state not in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        ):
            return

        # Derive elapsed from Optuna's own timestamps (thread-safe, no shared state)
        if trial.datetime_start and trial.datetime_complete:
            elapsed = (trial.datetime_complete - trial.datetime_start).total_seconds()
        else:
            elapsed = 0.0

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.datetime_start and t.datetime_complete]
        times = [(t.datetime_complete - t.datetime_start).total_seconds() for t in completed]
        avg = np.mean(times) if times else 0.0
        rem = avg * max(N_TRIALS - len(times), 0)

        with self._lock:
            try:
                best = study.best_value
            except ValueError:
                return
            if self._best_value is None or best < self._best_value:
                improvement = 0.0 if self._best_value is None else self._best_value - best
                self._best_value   = best
                self._trials_since = 0
                if improvement > 0:
                    print(f'  New best: {best:.6f}  (Δ={improvement:.6f})')
            else:
                self._trials_since += 1
                print(f'  No improvement for {self._trials_since} trial(s)')

        print(f'  Trial {trial.number}: {timedelta(seconds=int(elapsed))}  '
              f'avg: {timedelta(seconds=int(avg))}  '
              f'est. remaining: {timedelta(seconds=int(rem))}')


# ── Optuna objective ───────────────────────────────────────────
def make_objective(X_train, y_train, train_weights):

    def objective(trial):
        device  = gpu_for_trial(trial.number)
        max_bin = trial.suggest_int('max_bin', 64, 256, step=64)
        params = {
            'objective'        : 'multi:softprob',
            'num_class'        : NUM_CLASSES,
            'eval_metric'      : 'mlogloss',
            'tree_method'      : 'hist',
            'device'           : device,
            'nthread'          : NTHREAD_PER_TRIAL,
            'seed'             : SEED,
            'max_depth'        : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight' : trial.suggest_int('min_child_weight', 5, 50),
            'subsample'        : trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'gamma'            : trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_bin'          : max_bin,
        }

        skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_losses = []
        best_iters  = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            t_fold = time.time()
            try:
                dtrain = xgb.QuantileDMatrix(
                    X_train[tr_idx], label=y_train[tr_idx],
                    weight=train_weights[tr_idx],
                    nthread=NTHREAD_PER_TRIAL, max_bin=max_bin,
                )
                dval = xgb.QuantileDMatrix(
                    X_train[va_idx], label=y_train[va_idx],
                    nthread=NTHREAD_PER_TRIAL, max_bin=max_bin, ref=dtrain,
                )
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=NUM_BOOST_ROUND,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
            except xgb.core.XGBoostError as e:
                if 'cudaErrorMemoryAllocation' in str(e) or 'out of memory' in str(e).lower():
                    del dtrain, dval
                    gc.collect()
                    raise optuna.TrialPruned()
                raise

            proba = model.predict(dval)
            ll    = log_loss(y_train[va_idx], proba, labels=list(range(NUM_CLASSES)))
            best_iters.append(model.best_iteration)
            fold_losses.append(ll)

            print(f'  [trial {trial.number} | gpu {device}] '
                  f'fold {fold+1}/{N_FOLDS}: log-loss={ll:.6f}  '
                  f'best_iter={model.best_iteration}  ({time.time()-t_fold:.1f}s)')

            trial.report(np.mean(fold_losses), fold)
            if trial.should_prune():
                del model, dtrain, dval, proba
                gc.collect()
                raise optuna.TrialPruned()

            del model, dtrain, dval, proba
            gc.collect()

        mean_ll       = float(np.mean(fold_losses))
        avg_best_iter = int(np.mean(best_iters))
        trial.set_user_attr('avg_best_iteration', avg_best_iter)
        print(f'  [trial {trial.number}] mean log-loss={mean_ll:.6f} '
              f'avg_best_iter={avg_best_iter} (±{np.std(fold_losses):.6f})')
        return mean_ll

    return objective


# ── main ───────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    rng        = np.random.default_rng(SEED)
    load_start = time.time()

    # 1. Load
    print('Loading data...')
    X    = load_npz(FEAT_FILE)
    meta = np.load(META_FILE, allow_pickle=True)
    y    = meta['y'].astype(np.int32)
    feature_names = meta['feature_names']
    print(f'  X shape: {X.shape}  dtype: {X.dtype}')
    print_dist(y, 'Full dataset:')

    # 2. Stratified split — val/test keep full natural class distribution
    print('\nSplitting...')
    idx = np.arange(len(y))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y, test_size=TEST_FRAC, stratify=y, random_state=SEED,
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval, y_trainval,
        test_size=VAL_FRAC / (1 - TEST_FRAC),
        stratify=y_trainval, random_state=SEED,
    )
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    print_dist(y_val,  f'Val  (full dist, n={len(y_val):,}):')
    print_dist(y_test, f'Test (full dist, n={len(y_test):,}):')

    # 3. Downsample class 0 in training only
    print(f'\nDownsampling class 0 → {CLASS0_CAP:,}...')
    X_train, y_train = downsample_class0(X_train, y_train, CLASS0_CAP, rng)
    print_dist(y_train, f'Train (n={len(y_train):,}):')
    train_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    load_time = time.time() - load_start
    print(f'Data ready in {timedelta(seconds=int(load_time))}')
    print(f'CPU threads per trial: {NTHREAD_PER_TRIAL} × {N_JOBS} parallel = '
          f'{NTHREAD_PER_TRIAL * N_JOBS} total')

    # 4. Optuna HPO — 4 parallel trials, one per GPU
    print('\n' + '='*70)
    print(f'  Optuna HPO  ({N_TRIALS} trials, {N_JOBS} parallel)')
    print('='*70 + '\n')

    timing_cb = TimingCallback()
    study = optuna.create_study(
        direction  = 'minimize',
        study_name = 'xgboost_4class_optimization',
        sampler    = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
    )

    optuna_start = time.time()
    study.optimize(
        make_objective(X_train, y_train, train_weights),
        n_trials          = N_TRIALS,
        n_jobs            = N_JOBS,
        callbacks         = [timing_cb],
        catch             = (xgb.core.XGBoostError,),
        show_progress_bar = False,  # disabled: noisy with n_jobs > 1
    )
    optuna_time = time.time() - optuna_start

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f'\nHPO complete: {len(completed)} completed, {len(pruned)} pruned')
    print(f'Best log-loss: {study.best_value:.6f}')
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v:.6f}' if isinstance(v, float) else f'  {k}: {v}')

    joblib.dump(study, STUDY_OUT)
    print(f'Study saved → {STUDY_OUT}')

    # 5. Final model — best params, avg best_iteration from CV, val early stopping
    print('\n' + '='*70)
    print('  Training final model on cuda:0')
    print('='*70)

    best_params = {
        'objective'   : 'multi:softprob',
        'num_class'   : NUM_CLASSES,
        'eval_metric' : 'mlogloss',
        'tree_method' : 'hist',
        'device'      : gpu_for_trial(0),   # single GPU for final model
        'nthread'     : _cpu_count,
        'seed'        : SEED,
        **study.best_params,
    }
    avg_best_iter = study.best_trial.user_attrs.get('avg_best_iteration', NUM_BOOST_ROUND)
    final_rounds  = min(int(avg_best_iter * 1.1) + 20, NUM_BOOST_ROUND)
    print(f'  avg_best_iteration from HPO: {avg_best_iter} → final rounds cap: {final_rounds}')

    max_bin      = study.best_params.get('max_bin', 128)
    final_dtrain = xgb.QuantileDMatrix(
        X_train, label=y_train, weight=train_weights,
        nthread=_cpu_count, max_bin=max_bin,
    )
    final_dval = xgb.QuantileDMatrix(
        X_val, label=y_val, nthread=_cpu_count, max_bin=max_bin, ref=final_dtrain,
    )

    t_final = time.time()
    final_model = xgb.train(
        best_params, final_dtrain,
        num_boost_round=final_rounds,
        evals=[(final_dtrain, 'train'), (final_dval, 'val')],
        early_stopping_rounds=FINAL_EARLY_STOPPING,
        verbose_eval=50,
    )
    final_train_time = time.time() - t_final
    print(f'  Done in {timedelta(seconds=int(final_train_time))}  '
          f'best_iteration={final_model.best_iteration}')

    # 6. Evaluate
    print('\n' + '='*70)
    print('  Evaluation')
    print('='*70)
    eval_split(final_model, X_val,  y_val,  'Validation')
    test_proba, test_preds = eval_split(final_model, X_test, y_test, 'Test')

    # 7. Save model + test predictions
    final_model.save_model(MODEL_OUT)
    np.savez(RESULTS_OUT,
             y_true=y_test, y_pred=test_preds, y_proba=test_proba,
             enrolid=meta['enrolid'][idx_test])

    # 8. Feature importance
    print('\n── Top 20 features (weight importance) ──')
    importance = final_model.get_score(importance_type='weight')
    imp_list = [(feature_names[int(k[1:])], v)
                for k, v in importance.items()
                if int(k[1:]) < len(feature_names)]
    for name, score in sorted(imp_list, key=lambda x: x[1], reverse=True)[:20]:
        print(f'  {name}: {score:.0f}')

    # 9. Timing summary
    total_time = time.time() - load_start
    print('\n' + '='*70)
    print(f'  Data loading  : {timedelta(seconds=int(load_time))}')
    print(f'  Optuna HPO    : {timedelta(seconds=int(optuna_time))}')
    print(f'  Final training: {timedelta(seconds=int(final_train_time))}')
    print(f'  TOTAL         : {timedelta(seconds=int(total_time))}')
    print(f'\nOutputs in {DATA_DIR}:')
    print(f'  {os.path.basename(MODEL_OUT)}')
    print(f'  {os.path.basename(STUDY_OUT)}')
    print(f'  {os.path.basename(RESULTS_OUT)}')


if __name__ == '__main__':
    main()
