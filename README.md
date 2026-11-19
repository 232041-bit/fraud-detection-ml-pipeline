# Bank Account Fraud Detection – ML → Flask → Angular → GitHub CI/CD

Week 6 DevOps lab. A gradient-boosting model (scikit-learn `HistGradientBoostingClassifier` inside a fitted
`Pipeline`) trained on the Bank Account Fraud (BAF, NeurIPS 2022) base dataset (1,000,000 synthetic applications,
30 input features). Served by a Flask REST API, used by an Angular form, tested/built by GitHub Actions and the
static Angular UI deployed on GitHub Pages.

## Problem / task
Binary classification: `fraud_bool` (1 = fraud). Two fitted scikit-learn Pipelines (imputer + scaler + one-hot, then
Random Forest or Logistic Regression) are packaged into `backend/models/model_bundle.joblib` (the deep ensemble and MC-dropout network of Novelty 2 are in `backend/models/novelty_bundle.joblib`). The decision threshold of each
model is set for a 5 % false-positive rate on the unseen months 6-7.

## Model provenance
Models: `random_forest_fraud_model.pkl`, `logistic_regression_fraud_model.pkl` (trained by the project author with scikit-learn 1.9.1).
Metrics (months 6-7) are in `docs/model_metrics.json`: recall ≈ 0.46-0.47 at 5 % FPR, ROC-AUC ≈ 0.87, age predictive equality ≈ 0.38-0.40.
This is a teaching prototype, not a production-fair model.

## Research novelties (v2.0)
Two contributions extend the baseline. Full method, equations and results are in **[docs/NOVELTIES.md](docs/NOVELTIES.md)**.

| | Novelty 1: Drift-Aware Fraud Detection | Novelty 2: Uncertainty-Aware Fraud Detection |
|---|---|---|
| Problem | fixed 5 %-FPR threshold breaks under temporal drift | every case is auto-decided, even when the model is unsure |
| Method | PSI + KS monitoring of features and score, delayed-label feedback, hybrid quantile recalibration, optional per-age-group (fair) thresholds | deep ensemble, MC dropout, tree disagreement; predictive entropy and mutual information; APPROVE / MANUAL_REVIEW / FLAG_FRAUD |
| Code | `backend/novelties/drift.py` | `backend/novelties/uncertainty.py` |
| Key result | mean \|FPR − 5 %\| 2.41 % → 0.77 %; batches within ±1 pp 0 % → 80 %; predictive equality (age) 0.39 → 0.91 | automated errors −41 % (RF), −31 % (MC dropout), −20 % (deep ensemble) with a 4–7 % review rate |

New files:
- `backend/train_novelties.py` trains the deep ensemble and MC-dropout model, calibrates the policies and saves `backend/models/novelty_bundle.joblib`.
- `backend/experiments/run_experiments.py` reproduces every table and figure (`docs/novelty_results.json`, `docs/figures/*.png`).
- `backend/tests/test_novelties.py` unit-tests both modules and the new API.
- `data/stream_demo.csv` holds 1 500 real BAF rows per month for the live drift demo.

### API (Flask)
| Endpoint | Purpose |
|---|---|
| `POST /predict?model=random_forest\|deep_ensemble\|mc_dropout\|logistic_regression&fair=0\|1` | fraud probability, **decision** (APPROVE / MANUAL_REVIEW / FLAG_FRAUD), **uncertainty** (entropy, mutual information, std) and the **adaptive threshold** used |
| `POST /drift/batch?model=&fair=` | body `{"records": [...], "labels": [...]}` (labels optional) returns a PSI/KS drift report and recalibrates the threshold when needed |
| `GET /drift/status?model=` / `POST /drift/reset?model=` | current threshold state and history / restore the calibrated threshold |
| `GET /drift/demo-batch?month=0..7` | labelled demo batch for the dashboard |
| `GET /novelty/summary` | compact experiment results |

### Dashboard (Angular)
Three tabs: **Predict** (decision badge, risk bar with grey zone, uncertainty and threshold cards, fairness toggle),
**Drift monitor** (stream month-by-month batches, PSI/KS table, threshold timeline) and **Research results**
(all experiment tables and charts; this tab also works on GitHub Pages).

### Retrain the novelties (optional; artifacts are committed)
```powershell
git lfs pull
python -m pip install -r backend\requirements-research.txt
python backend\train_novelties.py
python backend\experiments\run_experiments.py
```

## Features
31 raw inputs (26 numeric incl. `month`, 5 categorical: payment_type, employment_status, housing_status, source, device_os).
See `frontend/src/app/feature-schema.ts`.

## Versions
Python 3.12, Node.js LTS (24), Angular CLI (latest). scikit-learn is pinned to 1.9.1 (the version the models were saved with).

## Install
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python backend\package_existing.py
npm install -g @angular/cli
.\setup_frontend.ps1        # only if frontend\ does not exist yet
```

## Run locally
Terminal 1 (Flask):  `python backend\app.py`  → http://127.0.0.1:5000/health
Terminal 2 (Angular): `cd frontend ; ng serve --open --proxy-config proxy.conf.json` → http://localhost:4200

## Tests
```powershell
python backend\test_pipeline.py
python -m pytest -q backend\tests
cd frontend ; npm run build
```

## Links
- Repository: https://github.com/232041-bit/fraud-detection-ml-pipeline
- GitHub Pages: https://232041-bit.github.io/fraud-detection-ml-pipeline/

## Deployment limitation
GitHub Pages hosts **only the static Angular UI**. Flask and the model run locally; the prediction button is disabled on Pages.

## Disclaimer
Research/teaching prototype trained on synthetic data. Not for real financial decisions.
