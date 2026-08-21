# Hip-fracture-in-diabetes paper — unified evaluation pipeline

Created 2026-06-15. Supersedes the scattered set of eval scripts/notebooks now
in `~/eval_archive_2026-06-15/old_scripts/`.

## Quick start

```bash
# regenerate all MarketScan figures + tables from existing prediction npz files
python run_full_eval.py

# also re-run UChicago BEHRT-LoRA inference (~10 min on GPU)
python run_full_eval.py --uchicago

# force re-run all model inference + everything (~30-40 min)
python run_full_eval.py --force-rerun

# subset
python run_full_eval.py --only marketscan --models behrt,xgb
```

## Layout

```
inference/        re-run model inference (slow, opt-in)
figures/          PR + ROC + CM plots
tables/           overall-macro, per-class AUC/AUPRC+CI, op-point, Brier
run_full_eval.py  CLI entry point

outputs/
  predictions/    canonical {y_true,y_pred,y_proba,enrolid} npz per (cohort, model)
  figs/           fig{1,2,4}_{behrt,xgboost,nn}.{pdf,png}
                  figUC_{A,B,C}_*.{pdf,png}
  tables/         table_{overall_macro,per_class_ci,operating_point,brier}.{csv,tex}
logs/             per-stage logs with ISO timestamps
```

## Reuse rules

By default inference is **skipped** for MarketScan — the existing prediction
npz files (XGB Jun-8, NN Jun-8, BEHRT-LoRA Jun-14) are the correct ones from
the clean-cohort retrains:

| model | source npz |
|---|---|
| XGBoost | `~/Testing_NN_XGBoost_on_4_risk_classes_clean_2026-06-08/xgboost_4class_results.npz` |
| NN | `~/Testing_NN_XGBoost_on_4_risk_classes_clean_2026-06-08/nn_4class_results.npz` |
| BEHRT-LoRA | `~/embeddings_BEHRT/behrt_4class_results_2026-06-08.npz` |
| UChicago BEHRT (stale, pre-BI_181-cleanup) | `~/embeddings_BEHRT/uchicago_output/uchicago_frax_behrt_results_2026-05-22.npz` |

UChicago BEHRT inference **does** re-run (the 05-22 npz is from the
pre-rebuild contaminated model). Output: `outputs/predictions/uchicago_behrt.npz`.

## Output → main.tex mapping

`main.tex` includes figures from `figures/overleaf_figures/`. Copy these
outputs over after each run:

| main.tex include | this pipeline produces |
|---|---|
| fig1_pr_{behrt,xgboost,nn}.pdf | outputs/figs/fig1_pr_{behrt,xgboost,nn}.{pdf,png} |
| fig2_roc_{behrt,xgboost,nn}.pdf | outputs/figs/fig2_roc_{behrt,xgboost,nn}.{pdf,png} |
| fig4_cm_{behrt,xgboost,nn}.pdf | outputs/figs/fig4_cm_{behrt,xgboost,nn}.{pdf,png} |
| figUC_A_fracture_rate_decile_2026-05-22.pdf | outputs/figs/figUC_A_fracture_rate_decile.pdf |
| figUC_B_roc_behrt_frax_2026-05-22.pdf | outputs/figs/figUC_B_roc_behrt_frax.pdf |
| figUC_C_pr_behrt_frax_2026-05-22.pdf | outputs/figs/figUC_C_pr_behrt_frax.pdf |
| Table 1 (overall macro, MarketScan) | outputs/tables/table_overall_macro.{csv,tex} |
| Table 3 (per-class AUC/AUPRC + CI) | outputs/tables/table_per_class_ci.{csv,tex} |
| Table 6 (op-point sens/PPV/NPV) | outputs/tables/table_operating_point.{csv,tex} |
| Table 11 (Brier raw + iso) | outputs/tables/table_brier.{csv,tex} |

AoU tables (2, 5, 7, 12) and figures (`*_aou.pdf`) are produced by a
**separate** workflow in the AoU workbench. Out of scope here.

## Encoding (CANONICAL — same across BEHRT, XGB, NN)

```
class 0 = >10 years (control)
class 1 = 1-5 years
class 2 = <1 year
class 3 = 5-10 years
```

Display order in figures (ascending duration): `[<1y, 1-5y, 5-10y, >10y]`
i.e. native indices `[2, 1, 3, 0]`.

## Reproducibility

- Bootstrap: 200 iter, `np.random.default_rng(35)`, shared across models
- BEHRT-LoRA must be LAST in the model dict so the XGB/NN CI values stay
  byte-identical to the poster CSV (otherwise the shared RNG state differs)
- Isotonic-calibrated Brier: 70/30 split, seed=42, stratified
- All paths absolute. No `cd`-dependent behaviour.
