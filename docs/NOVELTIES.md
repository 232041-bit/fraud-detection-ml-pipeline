# Research novelties: drift-aware and uncertainty-aware fraud detection

This document describes the two contributions added on top of the baseline BAF pipeline
(Random Forest / Logistic Regression, 5 % false-positive-rate operating point, chronological split).
It is the reference for the methodology and results sections of the extended paper. All numbers
below come from `docs/novelty_results.json`, produced by `backend/experiments/run_experiments.py`.

Baseline paper: Nasif, Jahin & Mridha, *Reinforcement-Guided Hyper-Heuristic Hyperparameter Optimization
for Fair and Explainable Spiking Neural Network-Based Financial Fraud Detection* (Knowledge-Based Systems, 2026).

---

## Novelty 1: Drift-Aware Fraud Detection Module

### Gap in the existing work
The baseline evaluates the model on future months (temporal split) but uses **one fixed decision
threshold**, chosen offline. The BAF data drifts strongly over time. For example, `velocity_4w` has a PSI of
3.0–5.2 relative to month 2, and the fraud rate rises from 0.87 % (month 2) to 1.47 % (month 7). As a result,
the realised FPR leaves the 5 % business constraint and nothing detects or corrects it.

### Method (`backend/novelties/drift.py`)
1. **Monitoring.** `DriftMonitor` stores a reference distribution for every input feature (decile bins plus
   a 5 000-value sample) and for the model score. Each incoming micro-batch is tested with
   * the Population Stability Index, PSI = Σ_b (a_b − e_b) · ln(a_b / e_b). The bands are stable < 0.10,
     moderate 0.10–0.25 and major ≥ 0.25.
   * the two-sample Kolmogorov–Smirnov test. A feature counts as drifted when D ≥ 0.10 and p < 0.01.

   Drift is declared when at least 3 features drift, any feature has PSI ≥ 0.25, or the score distribution drifts.
   The time index `month` is excluded from monitoring.
2. **Performance check.** When delayed fraud labels arrive (chargeback delay), the realised FPR and recall of the
   current policy are computed. A deviation of more than ±1 pp from 5 % also triggers recalibration.
3. **Threshold recalibration (hybrid quantile estimator).** If π is the prevalence and r the recall at the
   5 %-FPR point, then the share of all traffic above the threshold is
   a = 0.05·(1 − π) + r·π. π and r are estimated from the most recent labelled batches. The threshold is the
   (1 − a) quantile of the most recent *unlabelled* scores, which already reflect the drift. This estimate is
   blended with the label-based negative quantile once enough labelled negatives exist.
4. **Fairness-aware mode.** The same estimator runs separately for each age group (< 50, ≥ 50), so every group
   is held at 5 % FPR. This gives predictive equality by construction.
5. **Re-anchoring.** After a recalibration, the monitor's reference is moved to the data used for it. Later
   alerts therefore measure *new* change instead of re-firing on the same old shift.

### Experiment 1: long stream (months 3–7)
HistGradientBoosting is trained on months 0–1, and its threshold (5 % FPR) is fixed on month 2. Months 3–7
arrive as 20 micro-batches (4 per month). Labels are delayed by one batch, and the threshold for batch *b* only
uses information from batches before *b*.

| Policy | Mean \|FPR − 5 %\| | Batches within ±1 pp | Max FPR | Mean recall | Predictive equality (age) | Recalibrations |
|---|---|---|---|---|---|---|
| Static threshold (baseline) | 2.41 % | 0 % | 9.33 % | 59.9 %* | 0.387 | 0 |
| **Drift-aware (Novelty 1)** | **0.87 %** | **80 %** | 8.55 % | 51.7 % | 0.364 | **6** |
| **Drift-aware + fair** | **0.77 %** | **80 %** | 8.54 % | 48.4 % | **0.906** | 6 |
| Recalibrate every batch (ablation) | 0.78 % | 80 % | 8.55 % | 51.5 % | 0.362 | 18 |

\*The static recall is higher only because that policy *violates* the constraint (FPR ≈ 6–9 %).
The two batches above 8 % FPR are the first two in the stream, before any labels have arrived.

Drift-triggered recalibration matches the FPR control of recalibrating every batch while using
**3× fewer recalibrations** (6 vs 18). Fair mode raises predictive equality from 0.39 to 0.91.

### Experiment 1b: deployed Random Forest, month 6 → month 7
| Policy | Mean \|FPR − 5 %\| | Mean FPR | Mean recall | PE (age) |
|---|---|---|---|---|
| Static (baseline) | 2.03 % | 2.97 % | 37.0 % | 0.371 |
| **Drift-aware** | **0.71 %** | 4.67 % | **47.0 %** | 0.395 |
| **Drift-aware + fair** | **0.63 %** | 4.74 % | 45.0 % | **0.878** |

Here the drift *lowers* the realised FPR of the fixed threshold to 3 %, which wastes recall. Recalibration moves
the operating point back to 5 % and recovers 10 pp of recall.

Observation: input features drift heavily, but the score PSI stays small (≤ 0.05). The FPR still moves, because
both the prevalence and the score–label relationship shift. Monitoring the score alone is therefore not enough.
This is why the module combines feature PSI/KS, score PSI/KS and label feedback.

Figure: `docs/figures/drift_fpr_stream.png`.

---

## Novelty 2: Uncertainty-Aware Fraud Detection

### Gap in the existing work
Every prediction is forced into Fraud / Legitimate, including cases the model is unsure about.
In high-stakes finance these cases should go to a human reviewer instead of being decided automatically.

### Method (`backend/novelties/uncertainty.py`)
Each scorer produces M stochastic predictions p_1 … p_M:

| Estimator | Members |
|---|---|
| Random Forest (deployed) | 120 individual trees (tree disagreement) |
| **Deep ensemble** | 5 MLPs (64-32), each with a different initialisation and a different bagged negative sample |
| **Monte Carlo dropout** | numpy MLP (64-32, dropout 0.2) trained with dropout; T = 30 stochastic passes at inference |

Uncertainty measures: mean p̄, **predictive entropy** H[p̄] (total), expected entropy E_m H[p_m] (aleatoric),
**mutual information** H[p̄] − E_m H[p_m] (epistemic, BALD) and the member standard deviation.

**Three-way policy (`ThreeWayPolicy`)**, calibrated on month 6:
* FLAG_FRAUD: p̄ ≥ t_fraud (the 5 %-FPR threshold) **and** epistemic uncertainty ≤ u_max
* APPROVE: p̄ < t_low **and** epistemic uncertainty ≤ u_max
* MANUAL_REVIEW: everything else, i.e. the grey zone [t_low, t_fraud) (3 % of traffic) and the top 2 % by
  epistemic uncertainty

Automatic flags are a subset of {p̄ ≥ t_fraud}, so the automated FPR never exceeds the binary FPR.

### Experiment 2: calibration on month 6, test on the untouched month 7
| Model | AUC | Automated errors (binary → 3-way) | Error reduction | Auto FPR | Review rate | Recall (auto + review) |
|---|---|---|---|---|---|---|
| Random Forest | 0.869 | 3 736 → 2 193 | **−41.3 %** | 2.97 % → 1.56 % | 4.2 % | 37.0 % → 50.4 % |
| Deep Ensemble | 0.883 | 6 902 → 5 522 | **−20.0 %** | 6.62 % → 5.33 % | 6.5 % | 58.9 % → 69.6 % |
| MC Dropout | 0.888 | 5 368 → 3 716 | **−30.8 %** | 4.91 % → 3.38 % | 6.8 % | 52.4 % → 65.5 % |

Reviewers are assumed to resolve referred cases correctly. Under that assumption, the share of fraud that is either
flagged or referred rises by 11–14 pp.

**Does uncertainty identify errors?** Binary error rate by epistemic-uncertainty decile (decile 1 = most certain):

| Model | D1 | D5 | D8 | D9 | D10 |
|---|---|---|---|---|---|
| Random Forest | 0.1 % | 0.4 % | 1.0 % | 2.9 % | **32.6 %** |
| Deep Ensemble | 0.1 % | 1.4 % | 11.1 % | 18.6 % | **30.9 %** |
| MC Dropout | 0.1 % | 1.5 % | 6.8 % | 11.6 % | **27.8 %** |

**Ablation (mutual information).** Referring only epistemic-uncertainty cases gives the lowest automated error.
For RF it drops from 3.86 % to 1.99 % at a 2.9 % review rate. Adding the grey zone increases the recall captured
by review. Results for predictive entropy and the standard deviation are in `novelty_results.json`
(`uncertainty.*.ablations`).

Figure: `docs/figures/uncertainty_tradeoff.png` shows automated error versus review budget (0–12 %).

### Experiment 3: both novelties together (month 7 stream)
The three-way policy runs with the drift-adapted t_fraud:

| Model | Threshold | Auto FPR | Auto error | Review | Recall (auto + review) |
|---|---|---|---|---|---|
| Deep Ensemble | static | **5.33 % (violates 5 %)** | 6.10 % | 6.5 % | 69.6 % |
| Deep Ensemble | drift-aware | **4.42 %** | 5.16 % | 6.4 % | 67.5 % |
| Random Forest | static | 1.56 % | 2.36 % | 4.2 % | 50.4 % |
| Random Forest | drift-aware | 2.91 % | 3.67 % | 5.1 % | **57.8 %** |
| MC Dropout | static | 3.42 % | 4.15 % | 6.8 % | 66.3 % |
| MC Dropout | drift-aware | 3.48 % | 4.22 % | 6.8 % | 66.4 % |

The two modules complement each other. Novelty 1 keeps the automated FPR under the 5 % constraint after drift
(Deep Ensemble), and Novelty 2 keeps the automated error rate low by referring uncertain cases.

---

## Limitations (state these honestly in the paper)
* Reviewer accuracy is assumed to be perfect. The review queue (4–7 % of traffic) has a real cost.
* Recalibration needs at least one batch of delayed labels. The first batches after deployment are uncorrected.
* BAF rows carry no within-month timestamp, so micro-batches are random splits within each month.
* The fairness-aware mode equalises FPR (predictive equality) across two age groups only.
* The new uncertainty models are compact MLPs. They are not the spiking CSNPC network of the paper, but the
  module is model-agnostic: any scorer that can return member probabilities works.

## Reproduce
```bash
python -m pip install -r backend/requirements.txt matplotlib
git lfs pull                                   # full BAF csv (FraudDetection_Dataset.csv)
python backend/train_novelties.py              # ~1 min -> backend/models/novelty_bundle.joblib
python backend/experiments/run_experiments.py  # ~1 min -> docs/novelty_results.json, figures, summary
python -m pytest -q backend/tests
```
