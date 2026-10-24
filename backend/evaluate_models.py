r"""Evaluate YOUR two trained pipelines on the BAF test months (6-7) and write
backend/models/model_meta.json (5%-FPR thresholds, categories, ranges, features)
plus docs/model_metrics.json and the sample inputs.

Run from project root (needs data/FraudDetection_Dataset.csv):
    python backend\evaluate_models.py
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "random_forest": ROOT / "backend/models/random_forest_fraud_model.pkl",
    "logistic_regression": ROOT / "backend/models/logistic_regression_fraud_model.pkl",
}
TARGET = "fraud_bool"
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

df = pd.read_csv(ROOT / "data/FraudDetection_Dataset.csv")
features = [c for c in df.columns if c != TARGET]          # 31 raw inputs (incl. month)
numeric = [c for c in features if c not in CATEGORICAL]
test = df[df.month >= 6].reset_index(drop=True)           # months 6-7 = unseen future data
y = test[TARGET].to_numpy()
old = (test.customer_age >= 50).to_numpy()

meta = {"features": features,
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in numeric},
        "thresholds": {}, "metrics": {}}
probs = {}
for name, path in MODELS.items():
    pipe = joblib.load(path)
    p = pipe.predict_proba(test[features])[:, 1]
    probs[name] = p
    thr = float(np.quantile(p[y == 0], 0.95))              # threshold for 5% FPR
    pred = p >= thr
    f_old, f_young = pred[(y == 0) & old].mean(), pred[(y == 0) & ~old].mean()
    pe = float(min(f_old, f_young) / max(f_old, f_young)) if max(f_old, f_young) else 0.0
    meta["thresholds"][name] = thr
    meta["metrics"][name] = {"recall_at_5pct_fpr": float(pred[y == 1].mean()),
                             "fpr": float(pred[y == 0].mean()),
                             "roc_auc": float(roc_auc_score(y, p)),
                             "predictive_equality_age": pe}
    print(name, json.dumps(meta["metrics"][name]), "threshold", round(thr, 4))

(ROOT / "backend/models/model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/model_metrics.json").write_text(json.dumps(meta["metrics"], indent=2), encoding="utf-8")

# non-sensitive (synthetic) sample rows
rf, lr = probs["random_forest"], probs["logistic_regression"]
t_rf, t_lr = meta["thresholds"]["random_forest"], meta["thresholds"]["logistic_regression"]
legit = np.where((y == 0) & (rf < t_rf * 0.5) & (lr < t_lr))[0][0]
fraud_idx = np.where((y == 1) & (rf >= t_rf) & (lr >= t_lr))[0]
fraud = fraud_idx[np.argsort(-rf[fraud_idx])][20]
test.loc[[legit], features].to_csv(ROOT / "data/sample_inputs.csv", index=False)
test.loc[[fraud], features].to_csv(ROOT / "data/sample_fraud.csv", index=False)
print("Saved model_meta.json, docs/model_metrics.json, data/sample_*.csv")
