r"""Lab Manual 4.3: package YOUR two existing fitted pipelines into one bundle.
Run from project root:  python backend\package_existing.py
"""
from pathlib import Path
import json
import joblib

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SOURCES = {                      # trusted artifacts created by you
    "random_forest": MODELS / "random_forest_fraud_model.pkl",
    "logistic_regression": MODELS / "logistic_regression_fraud_model.pkl",
}
DEFAULT_MODEL = "random_forest"
TASK = "classification"
meta = json.loads((MODELS / "model_meta.json").read_text(encoding="utf-8"))
FEATURES = meta["features"]

pipelines = {}
for name, src in SOURCES.items():
    pipe = joblib.load(src)      # trusted artifact only
    assert hasattr(pipe, "predict_proba"), f"{name} must implement predict_proba"
    assert list(pipe.feature_names_in_) == FEATURES, f"{name}: feature order mismatch"
    pipelines[name] = pipe

bundle = {"pipeline": pipelines[DEFAULT_MODEL],       # default model (manual key)
          "pipelines": pipelines,                      # both models
          "default_model": DEFAULT_MODEL,
          "features": FEATURES, "task": TASK, "model_version": "1.0",
          "thresholds": meta["thresholds"],
          "categories": meta["categories"], "ranges": meta["ranges"],
          "classes": {"0": "Legitimate", "1": "Fraud"}}
joblib.dump(bundle, MODELS / "model_bundle.joblib", compress=3)
print("Saved model bundle:", len(FEATURES), "features,", list(pipelines))
