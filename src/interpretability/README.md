# BEHRT-LoRA — interpretability + clinical-utility analyses (npj revision)

Three analyses for the npj Digital Medicine revision:

1. **Occlusion** — "what if never diagnosed with X" → per-diagnosis risk drivers.
2. **Attention rollout** — what the model reads (contextual cross-check).
3. **Decision-curve analysis** — clinical utility / net benefit vs FRAX.

(1) and (2) are the interpretability pair — the BEHRT analog of the XGBoost
feature-importance story, supporting the deployment argument that a clinician
*without the full chart* can see why a patient was flagged. (3) answers the
separate "is acting on it worthwhile?" question. The interpretability methods are
described first; the decision curve is section 3.

Both run on the **University of Chicago cohort** (the same patients used for the
FRAX head-to-head), using the post-BI_181-rebuild LoRA checkpoint
`embeddings_BEHRT/output/behrt_lora_best.pt`. Predicted **hip-fracture risk** is
defined as `1 - P(control)` (native class 0 = control), matching the paper's
`uc_risk_score`.

## Environment

Use the `ml` conda env (has torch 2.5.1 + CUDA, pandas, peft, transformers 5.3):

```bash
python run_occlusion.py
python run_attention_rollout.py
python run_decision_curve.py   # fast, no model
```

Common flags: `--limit N` (first N patients, 0 = all), `--cases-only` (only the
hip-fracture cases — fast, good for case studies), `--top-k K` (per-patient
drivers kept). Outputs land in `outputs/`.

## Files

| file | what it does |
|------|--------------|
| `common.py` | rebuilds the model + loads UChicago sequences **exactly** as the eval pipeline; shared scoring helpers. Do not diverge from this. |
| `run_occlusion.py` | diagnosis-level occlusion ("what if never diagnosed with X"). |
| `run_attention_rollout.py` | attention rollout (what the model attends to). |
| `run_decision_curve.py` | decision-curve / net-benefit, BEHRT vs FRAX (no model needed). |
| `plot_calibration_comparison.py` | overlays raw vs recalibrated BEHRT on one decision curve (the "why calibration matters" figure). |
| `outputs/` | CSVs + figures (regenerated on each run). |

---

## 1. Occlusion — `run_occlusion.py`

**Question:** *"What if this patient had never been diagnosed with condition X?"*
For each patient and each distinct diagnosis in their record, we delete **every
occurrence** of that diagnosis (across all visits), re-score the patient, and
record the change in risk:

```
delta_X = risk(full record) - risk(record without X)
delta_X > 0  ->  X was pushing the patient's risk UP
delta_X < 0  ->  X was pushing it DOWN
```

**Outputs**
- `occlusion_global_importance.csv` — the **XGBoost-importance analog**: per
  diagnosis, the mean signed effect, mean |effect|, number of patients who have
  it, and how often it is a patient's single largest driver. Ranked by mean
  |effect|.
- `occlusion_per_patient.csv` — each patient's baseline risk, predicted class,
  and top-K diagnosis drivers (for "why was this patient flagged").
- `occlusion_long.csv` — the full (patient × diagnosis) effect table.
- `occlusion_global_importance.{png,pdf}` — top global drivers bar chart.

**Strength:** intuitive and clinician-readable ("removing osteoporosis dropped
this patient's risk by 0.22"). It is a **context-aware** counterfactual — when a
diagnosis is removed the transformer still re-reads the entire remaining record.

## 2. Attention rollout — `run_attention_rollout.py`

**Question:** *"Which diagnoses does the model actually look at?"* Attention
rollout (Abnar & Zuidema, 2020) traces information flow from each input token up
through every transformer layer to the `[CLS]` token that feeds the classifier:
average attention over heads, add the residual identity, row-normalise, and
multiply the per-layer matrices. The `[CLS]` row of the product is how much of
the final representation traces back to each diagnosis.

> Implementation note: transformers v5 no longer threads `output_attentions`
> through the encoder, so `AttentionCatcher` recomputes
> `softmax(QKᵀ/√d + mask)` via forward hooks on each self-attention module —
> identical to the eager path, version-independent.

**Outputs:** `attention_global_importance.csv`, `attention_per_patient.csv`,
`attention_long.csv`, `attention_global_importance.{png,pdf}` (same shapes as the
occlusion outputs, with attention fractions instead of risk deltas).

**Strength:** shows the model's contextual reading directly. **Caveat:**
attention-as-explanation is debated; use it to *corroborate* occlusion, not as
the sole evidence.

## 3. Decision-curve analysis — `run_decision_curve.py`

**Question:** *"Is acting on the score clinically worthwhile, and over what risk
thresholds?"* Net benefit weighs true positives against false positives, with the
false-positive penalty set by the odds of the threshold probability `p_t`:

```
NB(p_t) = TP/N - (FP/N) * (p_t / (1 - p_t))
```

Compares BEHRT-LoRA, FRAX, screen-all, and screen-none on the UChicago FRAX
subset (N=12,325; **71 hip fractures**, 0.58%). Reads the arrays already in
`eval_pipeline_2026-06-15/outputs/predictions/uchicago_behrt.npz` — **no model
run needed**. Patient-level bootstrap (seed 35) gives 95% CI bands and a
BEHRT-minus-FRAX difference band.

### Calibration matters here (read this)

DCA assumes the x-axis score is a **calibrated risk**. BEHRT's raw score
`1 - P(control)` is a *ranking* score and runs up to ~1.0, so thresholding it as a
probability floods the screen-positive group with false positives — on raw scores
BEHRT's net benefit goes **negative** and FRAX wins, even though BEHRT's AUROC
(0.884) far exceeds FRAX's (0.765). That is the classic DCA-on-an-uncalibrated-
score pitfall, not a real utility gap.

By **default** the script recalibrates *both* scores to the binary outcome with
**cross-validated isotonic regression** (5-fold, pooled out-of-fold, seed 35)
before computing net benefit — the standard, fair way to give a ranking tool a
decision curve, kept honest by out-of-fold pooling (no patient calibrated on
itself). `--raw` reproduces the uncalibrated curve; `--calib platt` switches to
parametric (logistic) recalibration. The conclusion is the same under either
method.

This is consistent with the paper's framing: BEHRT is a triage/ranking instrument
that must be recalibrated to the local population before deployment.

### Why FRAX is recalibrated too (not just BEHRT)

FRAX is **not** a random number borrowed from other patients — each FRAX score
was computed from *this* patient's own risk factors. The reason it still needs
recalibrating is a **mismatch of questions**, easiest to see as three steps:

1. **The question FRAX answers:** "Over the next **10 years**, what is the
   probability this patient has a hip fracture?" → e.g. *8%*. An honest answer to
   *FRAX's* question.
2. **The question the decision curve needs answered:** the threshold is applied
   against the **outcome we actually recorded** (`y_hip`) — "what is the
   probability this patient has a **recorded hip-fracture event in this study's
   observation window**?" Across the cohort that averages **0.58%**, because it is
   a shorter, observed window, not a 10-year projection.

   These are two different questions, so their answers sit on two different number
   scales. FRAX's "8%" answers question 1; the threshold `p_t = 5%` is a cutoff on
   the answer to question 2.
3. **Why mixing them breaks the threshold:** "flag everyone with risk ≥ 5%" means
   5% chance of the *study outcome* (Q2). Feed in FRAX's raw answer to Q1 and a
   patient FRAX rates "8%" clears the bar — even though their chance of the
   recorded outcome may be ~1%, well under 5%. Raw FRAX therefore over-flags
   low-risk patients → false positives → its net benefit looks worse than it
   should. The model isn't wrong; we compared an answer to Q1 against a threshold
   meant for Q2.

**What recalibration does:** it builds an order-preserving *translation table*
from "FRAX's answer to Q1" → "the answer to Q2," learned from the data (among
patients FRAX rated ~8%, what fraction actually had the recorded outcome? → say
1%, so 8% → 1%). Now FRAX's score is in the same units as the threshold and the
outcome, so thresholding compares like with like.

**Analogy:** FRAX reports in Fahrenheit; the threshold thermostat is set in
Celsius. The ordering of patients coldest→hottest is identical in either unit
(so discrimination / AUROC is unchanged — recalibration is monotonic), but you
must convert F→C before comparing a reading to the thermostat. Here the "unit"
gap is *10-year projected risk* vs *risk of the outcome as recorded in this
study's window*. Both scores go through the **identical** pipeline so the
head-to-head is fair.

> Alternative framing worth knowing: FRAX is an established instrument with its
> own validated treatment thresholds (~3% hip / 20% major fracture). Evaluating
> FRAX on its **native** scale at those thresholds (`--raw` for FRAX) answers a
> different, equally valid question — "how does FRAX perform *as clinically
> used*" — rather than the like-for-like discrimination+utility comparison the
> recalibrated curve gives.

**Result (recalibrated, isotonic default — Platt gives the same picture):** BEHRT
dominates FRAX, screen-all and screen-none across the whole screening range
(threshold ≈ 0.002–0.10). The BEHRT−FRAX net-benefit difference is positive with a
95% CI excluding zero at every clinically relevant threshold (≈ +0.003–0.004 ≈
**3–4 extra true hip-fracture cases identified per 1,000 patients screened at no
added false-positive cost vs FRAX**).

**Outputs** (`{suffix}` = `platt` | `isotonic` | `raw`):
- `decision_curve_{suffix}.csv` — net benefit + 95% CI for BEHRT, FRAX, treat-all,
  treat-none, and the BEHRT−FRAX difference, by threshold.
- `decision_curve_summary_{suffix}.csv` — the above at 0.5/1/2/3/5% thresholds.
- `decision_curve_{suffix}.{png,pdf}` — the decision curve with CI bands.

Suggested manuscript use: a main-text decision-curve figure (recalibrated) next
to the FRAX ROC/PR comparison, with one sentence noting the raw-vs-recalibrated
contrast as direct evidence for the ranking-tool framing.

---

## How to read the occlusion and attention views together (important)

Single-diagnosis occlusion measures each diagnosis's **marginal** effect with the
rest of the record held fixed. It is therefore blind to **redundancy** and
**interactions** — exactly what a transformer exploits:

- **Redundancy:** if two diagnoses carry overlapping signal (e.g. osteoporosis
  and a prior fragility fracture), removing *either one alone* barely moves the
  score, so occlusion under-rates both even though together they drive the call.
- **Interaction:** a diagnosis that only matters in combination with another
  shows little effect when removed in isolation.

So the two views answer different questions and should be reported together:

| | occlusion | attention rollout |
|---|---|---|
| asks | what *changes the prediction* if removed | what the model *reads* |
| captures | marginal causal effect | contextual focus |
| blind to | redundancy / interactions | true effect on the output |

Where they **agree**, the explanation is robust. Where they **disagree**,
attention often reveals redundancy that occlusion hides (e.g. the model attends
heavily to diabetes + its complications — near-universal in this cohort and thus
individually low-impact under occlusion — while occlusion isolates osteoporosis
as the decisive, non-redundant driver).

## Suggested use in the manuscript

- A global occlusion bar chart as the BEHRT counterpart to XGBoost feature
  importance (face-validity hook alongside Fig. 1).
- 2–3 per-patient driver vignettes (run with `--cases-only`) showing the model's
  top reasons for flagging real hip-fracture cases — the "clinician without the
  full chart" deployment point.
- One supplementary paragraph + the attention figure as the contextual
  cross-check, with the redundancy/interaction caveat stated honestly.
