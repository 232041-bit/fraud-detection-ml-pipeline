# Bank Account Fraud Detection – ML → Flask → Angular → GitHub CI/CD

Week 6 DevOps lab. A gradient-boosting model (scikit-learn `HistGradientBoostingClassifier` inside a fitted
`Pipeline`) trained on the Bank Account Fraud (BAF, NeurIPS 2022) base dataset (1,000,000 synthetic applications,
30 input features). Served by a Flask REST API, used by an Angular form, tested/built by GitHub Actions and the
static Angular UI deployed on GitHub Pages.

## Problem / task
Binary classification: `fraud_bool` (1 = fraud). Two fitted scikit-learn Pipelines (imputer + scaler + one-hot, then
Random Forest or Logistic Regression) are packaged into `backend/models/model_bundle.joblib`. The decision threshold of each
model is set for a 5 % false-positive rate on the unseen months 6-7.

## Model provenance
Models: `random_forest_fraud_model.pkl`, `logistic_regression_fraud_model.pkl` (trained by the project author with scikit-learn 1.9.1).
Metrics (months 6-7) are in `docs/model_metrics.json`: recall ≈ 0.46-0.47 at 5 % FPR, ROC-AUC ≈ 0.87, age predictive equality ≈ 0.38-0.40.
This is a teaching prototype, not a production-fair model.

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
