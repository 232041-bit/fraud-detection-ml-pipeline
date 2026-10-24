from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
PIPELINES = BUNDLE["pipelines"]
DEFAULT_MODEL = BUNDLE["default_model"]
THRESHOLDS = BUNDLE["thresholds"]          # decision thresholds at ~5% FPR
FEATURES = BUNDLE["features"]
TASK = BUNDLE.get("task", "classification")
CLASSES = BUNDLE["classes"]
CATEGORIES = BUNDLE.get("categories", {})

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   models=list(PIPELINES), default_model=DEFAULT_MODEL,
                   model_version=BUNDLE.get("model_version"))


@app.post("/predict")
def predict():
    name = request.args.get("model", DEFAULT_MODEL)
    if name not in PIPELINES:
        return jsonify(error="Unknown model", allowed=list(PIPELINES)), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch",
                       missing=missing, extra=extra), 400
    for col, allowed in CATEGORIES.items():
        if data[col] not in allowed:
            return jsonify(error=f"Invalid value for {col}",
                           allowed=allowed), 400
    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        # CUSTOM INFERENCE BLOCK: fraud probability + 5%-FPR business threshold
        fraud_prob = float(PIPELINES[name].predict_proba(frame)[0][1])
        threshold = float(THRESHOLDS[name])
        label = CLASSES["1"] if fraud_prob >= threshold else CLASSES["0"]
        return jsonify({"prediction": label, "task": TASK, "model": name,
                        "fraud_probability": fraud_prob, "threshold": threshold,
                        "probabilities": [1 - fraud_prob, fraud_prob],
                        "disclaimer": "Research/teaching prototype only. "
                                      "Trained on the synthetic BAF dataset."})
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
