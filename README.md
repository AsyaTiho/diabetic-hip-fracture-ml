# Prediction of Hip Fracture in Patients with Diabetes

Code accompanying the manuscript *"Prediction of Hip Fracture in Patients with
Diabetes"* (under review, *npj Digital Medicine*). It covers BEHRT-LoRA
sequence modelling of diagnosis histories, gradient-boosting and neural-network
baselines, evaluation, model interpretability, and external validation.

This repository contains **code and small reference files only**. No patient
data, trained model weights, intermediate arrays, or logs are included — see
[Data availability](#data-availability).

## Repository layout

```
src/                          analysis code (run in place; imports are flat)
  behrt/                      BEHRT MLM pretraining, LoRA fine-tuning, cohort/
                             label/sequence build, MarketScan label re-score
  baselines/                 XGBoost + sparse NN (4-class) with Optuna search
  evaluation/                shared inference + ROC/PR/confusion figures,
                             metric tables, Brier scores, operating points
  interpretability/          occlusion, attention rollout, decision-curve (DCA)
                             vs FRAX, no-prior-bone-disease subgroup test
  external_validation/aou/   All of Us rebuild (BEHRT/NN re-fit + evaluation)
  robustness/                temporal-leakage checks: landmark + matched cohort
                             builders, structure/age-gap cheaters, robustness figs
configs/                      per-arm env files (primary / landmark / matched)
notebooks/                    cohort/sequence build, UChicago prep, exploration
reference/phecode_maps/       public PheCode / PheCodeX maps + definitions
data/                         place datasets here (git-ignored; see data/README.md)
models/                       trained checkpoints land here (git-ignored)
results/                      generated figures/tables (git-ignored)
```

## How the cohort was built

The code that turns the raw MarketScan databases into `patient_sequences.pkl`,
`hip_frac_labels_CANONICAL.pkl` and the one-hot matrices is not shipped: it is
useless without the licensed source data and carries the database paths of the
machine it ran on. [`docs/COHORT_CONSTRUCTION.md`](docs/COHORT_CONSTRUCTION.md)
records what it does in full: the inclusion criteria and the SQL, the extraction
query and why it is shaped the way it is, how sequences and the vocabulary are
built, and how the labels are derived.

## Pipeline order

1. **Build cohort, labels, sequences** — `src/behrt/_build_canonical_labels.py`,
   then `notebooks/patient_seqs.ipynb` /
   `notebooks/building_dataset_for_hip_within_1_year.ipynb`, then
   `src/behrt/build_finetune_dataset.py`.
2. **Pretrain + fine-tune BEHRT** — `src/behrt/train_behrt_UPDATED.py` (MLM
   pretraining), then `src/behrt/finetune_behrt_lora.py` (LoRA 4-class head).
   The fine-tune always blanks patient age (temporal-leakage correction, see
   below), reproducing the age-blind model reported in the manuscript. Drive an
   arm with `python src/behrt/run_pipeline.py --config configs/primary.env`, or
   use `src/behrt/run_pipeline.sh`.
3. **Train baselines** — `src/baselines/train_xgboost_4class.py`,
   `src/baselines/train_nn_4class.py`.
4. **Evaluate** — `src/evaluation/run_full_eval.py` (drives `inference/`,
   `figures/`, `tables/`).
5. **Interpretability + clinical utility** — `src/interpretability/run_occlusion.py`,
   `run_attention_rollout.py`, `run_decision_curve.py`, `run_subgroup_no_bone.py`,
   then `make_importance_figures.py`.
6. **External validation** — `src/external_validation/aou/` (All of Us) and
   `notebooks/prepare_uchicago_for_behrt.ipynb` (UChicago prep).
7. **Robustness / temporal-leakage analysis** — `src/robustness/` builds the
   landmark and demographically matched sensitivity cohorts, fine-tunes them
   through the same script (`run_pipeline.py --config configs/landmark_180.env`
   or `configs/matched_180_k5.env --score`), and runs the structure-only and
   age-gap "cheater" tests plus the robustness figures. See
   `src/robustness/README.md`.

## Which code produces which manuscript result

| Manuscript item | Code |
| --- | --- |
| Overall ROC / PR / confusion (MarketScan, AoU) | `src/evaluation/figures/make_figures.py` via `run_full_eval.py` |
| Metric / Brier / operating-point tables | `src/evaluation/tables/make_tables.py` |
| FRAX comparison (ROC, PR) | `src/evaluation/figures/make_uchicago_figures.py` |
| Decision-curve analysis vs FRAX | `src/interpretability/run_decision_curve.py` |
| Occlusion + attention rankings | `src/interpretability/run_occlusion.py`, `run_attention_rollout.py`, `make_importance_figures.py` |
| No-prior-bone-disease subgroup | `src/interpretability/run_subgroup_no_bone.py` |
| Table 1 (cohort characteristics) | `notebooks/cohort_characteristics.ipynb` (aggregate output in `src/behrt/table1_cohort.csv`) |
| Robustness: landmark + matched arms | builders in `src/robustness/`, fine-tuned via `src/behrt/run_pipeline.py` with `configs/landmark_180.env` / `configs/matched_180_k5.env` |
| Robustness: structure-only + age-gap cheaters | `src/robustness/cheater_structure_age.py`, `src/robustness/cheater_shape_decompose.py` |
| Robustness figures (Discussion) | `src/robustness/fig_robustness.py`, `src/robustness/fig_landmark_attrition.py` |

Component folders keep their own notes where one exists
(`src/evaluation/README.md`, `src/interpretability/README.md`,
`src/robustness/README.md`).

## Temporal-leakage correction (age-blind model)

The risk classes are defined by the interval between the diabetes index and the
hip fracture, and a patient's age is deterministically related to that interval,
so an age embedding leaks the outcome **when case records are cut at the
fracture**. For that design, the primary reported model, `finetune_behrt_lora.py`
blanks every age token before training and inference.

The landmark and matched arms keep age. Cutting every record at a fixed interval
after the diabetes index makes the age gap identical for every patient, so it
carries no information about the outcome: a predictor given only the age gap
scores 0.505 there, against 0.903 on the cut-at-fracture design. Age is a genuine
hip-fracture risk factor, so withholding it where it does not leak would weaken
the model for nothing.

The channel is therefore selected per arm by `BEHRT_AGE_MODE`, which is
**required and has no default**: `removed` for the primary arm, `normal` for the
landmark and matched arms. Each file in `configs/` sets it, and the chosen mode
is recorded in the checkpoint and the prediction files, so a result states how it
was produced. The leak itself is demonstrated (not reproduced as a
usable model) by the age-gap cheater in `src/robustness/`, which shows that the
age gap alone separates the fracture classes on the original data and collapses
to chance once the observation window is fixed by the landmark cut.

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate hipfx
```

A CUDA GPU is required for BEHRT pretraining/fine-tuning. The published runs
used the versions pinned in `requirements.txt`.

## Configuring paths

Input/output locations are read from environment variables, each with a
repo-relative default. Set them to point at your local data/output copies, or
populate the default folders. Relative defaults resolve from the current working
directory, so either run scripts from the repo root or set absolute paths.

| Variable | Default | Holds |
| --- | --- | --- |
| `HIPFX_BEHRT_DIR` | `data/behrt` | BEHRT data: sequences, labels, vocab, result `.npz` |
| `HIPFX_BASELINES_DIR` | `data/baselines` | XGBoost/NN one-hot features, models, result `.npz` |
| `HIPFX_UCHICAGO_DIR` | `data/uchicago` | UChicago outcome / FRAX CSV |
| `HIPFX_RESULTS_DIR` | `results` | Pipeline outputs (`outputs/predictions`, `figs`, `tables`) |
| `HIPFX_BEHRT_CODE` | in-repo `src/behrt` | BEHRT model/dataset modules imported via `sys.path` |
| `HIPFX_MARKETSCAN_DEMO_DIR` | `data/marketscan_demographics` | Proprietary MarketScan demographic DBs (sex, birth year, enrollment visibility) used only by the matched-cohort matching in `src/robustness/` |

`HIPFX_BEHRT_CODE` defaults to the in-repo `src/behrt`, so the evaluation and
interpretability scripts import the shipped model definitions out of the box.
See `.env.example`. The notebooks under `notebooks/` use the same relative
defaults but set them inline (edit the path cell near the top).

The per-arm variables `BEHRT_FINETUNE_DATA`, `BEHRT_SPLIT_FILE`,
`BEHRT_OUTPUT_DIR`, `BEHRT_FINETUNE_CKPT`, and `BEHRT_PRED_OUT` select which
sequences, split, checkpoint, and prediction file a BEHRT run uses. They are set
by the files in `configs/` (one per arm) and read by `finetune_behrt_lora.py`
and the BEHRT inference, so you choose an arm by passing a config to
`run_pipeline.py` rather than exporting them by hand.

## Data availability

The clinical datasets cannot be redistributed and are **not** part of this
repository:

- **MarketScan** (Merative/IBM Commercial Claims) — licensed; available from
  Merative under a data-use agreement.
- **All of Us** — available to credentialed researchers through the All of Us
  Researcher Workbench (controlled tier).
- **UChicago** cohort — institutional EHR data, available only under the
  governing IRB / data-use agreement.

The `reference/phecode_maps/` tables are the publicly available PheCode and
PheCodeX resources, redistributed here for convenience.

Trained weights, patient sequences, prediction arrays, and run logs are omitted
by design; all are regenerated by running the pipeline on licensed data.

## Citation

See `CITATION.cff`. Licensed under the MIT License (`LICENSE`).
