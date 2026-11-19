"""Flask REST API for the BAF fraud-detection pipeline + the two research novelties.

Endpoints
  GET  /health                     service + model information
  POST /predict?model=&fair=       fraud score, uncertainty, three-way decision, adaptive threshold
  GET  /drift/status?model=        current drift-aware threshold state + history
  POST /drift/batch?model=&fair=   send a micro-batch (records [+ delayed labels]) -> PSI/KS
                                   drift report and (if needed) threshold recalibration
  POST /drift/reset?model=         restore the calibrated (static) threshold
  GET  /drift/demo-batch?month=    a labelled demo batch from data/stream_demo.csv
  GET  /novelty/summary            compact experiment results (docs/novelty_summary.json)
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
from novelties import (AdaptiveThresholdPolicy, DriftMonitor, ThreeWayPolicy,  # noqa: E402
                       TreeEnsembleUncertainty, age_group, uncertainty_metrics)

BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
PIPELINES = BUNDLE["pipelines"]
DEFAULT_MODEL = BUNDLE["default_model"]
THRESHOLDS = BUNDLE["thresholds"]          # original decision thresholds at ~5% FPR
FEATURES = BUNDLE["features"]
TASK = BUNDLE.get("task", "classification")
CLASSES = BUNDLE["classes"]
CATEGORIES = BUNDLE.get("categories", {})

# ----------------------------------------------------------------------------- novelties
NOVELTY = joblib.load(BASE / "models/novelty_bundle.joblib")
TARGET_FPR = NOVELTY["target_fpr"]
ESTIMATORS = {   # every model exposes member_probs(frame) -> (M, n)
    "random_forest": TreeEnsembleUncertainty(PIPELINES["random_forest"]),
    "deep_ensemble": NOVELTY["uncertainty_models"]["deep_ensemble"],
    "mc_dropout": NOVELTY["uncertainty_models"]["mc_dropout"],
}
MODEL_INFO = {
    "random_forest": {"label": "Random Forest", "uncertainty": "tree-ensemble disagreement"},
    "logistic_regression": {"label": "Logistic Regression", "uncertainty": "predictive entropy only"},
    "deep_ensemble": {"label": "Deep Ensemble (5 MLPs)", "uncertainty": "deep ensemble"},
    "mc_dropout": {"label": "MC-Dropout MLP", "uncertainty": "Monte Carlo dropout (T=30)"},
}
ALL_MODELS = list(MODEL_INFO)
THREE_WAY = {m: ThreeWayPolicy.from_params(NOVELTY["policies"][m]) for m in ALL_MODELS}
MONITORS = {m: DriftMonitor.from_dict(NOVELTY["drift_reference"][m]) for m in ALL_MODELS}
LOCK = threading.Lock()


def _new_policy(name: str, fair: bool = False) -> AdaptiveThresholdPolicy:
    return AdaptiveThresholdPolicy(NOVELTY["thresholds"][name], TARGET_FPR, fair=fair,
                                   mode="drift", group_thresholds=NOVELTY["group_thresholds"][name])


POLICIES = {m: _new_policy(m) for m in ALL_MODELS}
DRIFT_LOG: dict[str, list] = {m: [] for m in ALL_MODELS}
DECISION_LABEL = {0: "APPROVE", 1: "MANUAL_REVIEW", 2: "FLAG_FRAUD"}

app = Flask(__name__)


def _member_probs(name: str, frame: pd.DataFrame) -> np.ndarray:
    if name in ESTIMATORS:
        return ESTIMATORS[name].member_probs(frame)
    return PIPELINES[name].predict_proba(frame)[:, 1][None, :]


def _scores(name: str, frame: pd.DataFrame) -> dict:
    return uncertainty_metrics(_member_probs(name, frame))


def _validate(data) -> tuple[dict | None, tuple | None]:
    if not isinstance(data, dict):
        return None, (jsonify(error="Expected a JSON object"), 400)
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return None, (jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400)
    for col, allowed in CATEGORIES.items():
        if data[col] not in allowed:
            return None, (jsonify(error=f"Invalid value for {col}", allowed=allowed), 400)
    return data, None


def _model_arg():
    name = request.args.get("model", DEFAULT_MODEL)
    if name not in MODEL_INFO:
        return None, (jsonify(error="Unknown model", allowed=ALL_MODELS), 400)
    return name, None


def _fair_arg() -> bool:
    return request.args.get("fair", "0").lower() in ("1", "true", "yes")


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   models=ALL_MODELS, model_info=MODEL_INFO, default_model=DEFAULT_MODEL,
                   model_version=BUNDLE.get("model_version"),
                   novelty_version=NOVELTY.get("version"),
                   novelties=["drift-aware adaptive threshold (PSI/KS)",
                              "uncertainty-aware three-way decision"])


@app.post("/predict")
def predict():
    name, err = _model_arg()
    if err:
        return err
    data, err = _validate(request.get_json(silent=True))
    if err:
        return err
    fair = _fair_arg()
    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        members = _member_probs(name, frame)
        met = uncertainty_metrics(members)
        fraud_prob = float(met["mean"][0])
        group = int(age_group(frame.customer_age)[0])
        with LOCK:
            pol = POLICIES[name]
            adaptive = float(pol.group_thresholds[group] if fair else pol.threshold)
            pstate = pol.state()
        three = THREE_WAY[name]
        code = int(three.decide({k: v for k, v in met.items()}, t_fraud=adaptive)[0])
        t_low = three.t_low * adaptive / three.t_fraud
        label = CLASSES["1"] if fraud_prob >= adaptive else CLASSES["0"]
        reasons = []
        if met[three.key][0] > three.u_max:
            reasons.append("epistemic uncertainty above calibrated limit")
        if t_low <= fraud_prob < adaptive:
            reasons.append("score inside grey zone below the fraud threshold")
        return jsonify({
            "prediction": label, "task": TASK, "model": name,
            "fraud_probability": fraud_prob, "threshold": adaptive,
            "probabilities": [1 - fraud_prob, fraud_prob],
            # ---- Novelty 2: uncertainty-aware three-way decision
            "decision": DECISION_LABEL[code],
            "decision_reasons": reasons,
            "uncertainty": {
                "method": MODEL_INFO[name]["uncertainty"],
                "members": int(members.shape[0]),
                "std": float(met["std"][0]),
                "predictive_entropy": float(met["predictive_entropy"][0]),
                "aleatoric_entropy": float(met["aleatoric_entropy"][0]),
                "mutual_information": float(met["mutual_information"][0]),
                "epistemic_limit": three.u_max if np.isfinite(three.u_max) else None,
                "grey_zone": [t_low, adaptive],
            },
            # ---- Novelty 1: drift-aware threshold actually used
            "drift": {"adaptive_threshold": adaptive, "calibrated_threshold": pstate["base_threshold"],
                      "original_threshold": float(THRESHOLDS.get(name, pstate["base_threshold"])),
                      "fair_mode": fair, "age_group": ">=50" if group else "<50",
                      "recalibrations": pstate["recalibrations"]},
            "disclaimer": "Research/teaching prototype only. Trained on the synthetic BAF dataset."})
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


# ----------------------------------------------------------------------------- drift API
def _compact_report(rep: dict, top: int = 10) -> dict:
    feats = sorted(rep["features"].items(), key=lambda kv: -kv[1]["psi"])[:top]
    return {"n": rep["n"], "drift_detected": rep["drift_detected"], "severity": rep["severity"],
            "feature_drift": rep["feature_drift"], "score_drift": rep["score_drift"],
            "max_psi": rep["max_psi"], "drifted_features": rep["drifted_features"],
            "score": rep["score"],
            "top_features": [{"feature": c, **v} for c, v in feats]}


@app.get("/drift/status")
def drift_status():
    name, err = _model_arg()
    if err:
        return err
    with LOCK:
        return jsonify(model=name, policy=POLICIES[name].state(), log=DRIFT_LOG[name][-50:])


@app.post("/drift/reset")
def drift_reset():
    name, err = _model_arg()
    if err:
        return err
    with LOCK:
        POLICIES[name] = _new_policy(name, _fair_arg())
        MONITORS[name] = DriftMonitor.from_dict(NOVELTY["drift_reference"][name])
        DRIFT_LOG[name] = []
        return jsonify(model=name, policy=POLICIES[name].state())


@app.post("/drift/batch")
def drift_batch():
    name, err = _model_arg()
    if err:
        return err
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("records"), list) or not body["records"]:
        return jsonify(error='Expected {"records": [ {...}, ... ], "labels": [0/1, ...] (optional)}'), 400
    records, labels = body["records"], body.get("labels")
    if len(records) > 20000:
        return jsonify(error="Batch too large (max 20000 records)"), 400
    for i, rec in enumerate(records):
        _, e = _validate(rec)
        if e:
            return jsonify(error=f"Record {i}: invalid", detail=e[0].get_json()), 400
    if labels is not None and (len(labels) != len(records) or
                               any(v not in (0, 1) for v in labels)):
        return jsonify(error="labels must be a 0/1 list with one entry per record"), 400
    frame = pd.DataFrame([[r[f] for f in FEATURES] for r in records], columns=FEATURES)
    try:
        met = _scores(name, frame)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    scores, groups = met["mean"], age_group(frame.customer_age)
    report = MONITORS[name].check(frame, scores)
    fair = _fair_arg()
    with LOCK:
        pol = POLICIES[name]
        if pol.fair != fair:
            pol.fair = fair
        before = pol.state()
        perf_before = None
        feedback = None
        if labels is not None:
            y = np.asarray(labels, int)
            pred = pol.decide(scores, groups)
            neg = y == 0
            perf_before = {"fpr": float(pred[neg].mean()) if neg.any() else None,
                           "recall": float(pred[~neg].mean()) if (~neg).any() else None}
        pol.observe_scores(scores, groups)
        if labels is not None:
            feedback = pol.observe_labels(scores, groups, labels)
        rec = pol.update(report, feedback)
        if rec["recalibrated"]:
            MONITORS[name].rebase(frame, scores)      # new alerts = change since adaptation
        perf_after = None
        if labels is not None:
            y = np.asarray(labels, int)
            pred = pol.decide(scores, groups)
            neg = y == 0
            perf_after = {"fpr": float(pred[neg].mean()) if neg.any() else None,
                          "recall": float(pred[~neg].mean()) if (~neg).any() else None}
        codes = THREE_WAY[name].decide(met, t_fraud=pol.threshold_for(groups))
        entry = {"batch": len(DRIFT_LOG[name]) + 1, "n": len(records),
                 "month": body.get("month"),
                 "drift_detected": report["drift_detected"], "severity": report["severity"],
                 "max_psi": report["max_psi"],
                 "score_psi": report["score"]["psi"] if report["score"] else None,
                 "threshold_before": before["threshold"], "threshold_after": pol.threshold,
                 "group_thresholds": {str(k): v for k, v in pol.group_thresholds.items()},
                 "recalibrated": rec["recalibrated"], "reason": rec["reason"],
                 "performance_before": perf_before, "performance_after": perf_after,
                 "decisions": {DECISION_LABEL[c]: int((codes == c).sum()) for c in (0, 1, 2)}}
        DRIFT_LOG[name].append(entry)
        return jsonify(model=name, report=_compact_report(report), update=entry,
                       policy=pol.state())


_DEMO = None


@app.get("/drift/demo-batch")
def demo_batch():
    global _DEMO
    if _DEMO is None:
        path = ROOT / "data/stream_demo.csv"
        if not path.exists():
            return jsonify(error="data/stream_demo.csv not found"), 404
        _DEMO = pd.read_csv(path)
    try:
        month = int(request.args.get("month", 7))
        n = min(int(request.args.get("n", 1500)), 1500)
    except ValueError:
        return jsonify(error="month and n must be integers"), 400
    part = _DEMO[_DEMO.month == month].head(n)
    if part.empty:
        return jsonify(error="month must be between 0 and 7"), 400
    records = json.loads(part[FEATURES].to_json(orient="records"))
    return jsonify(month=month, records=records, labels=part["fraud_bool"].astype(int).tolist())


@app.get("/novelty/summary")
def novelty_summary():
    path = ROOT / "docs/novelty_summary.json"
    if not path.exists():
        return jsonify(error="Run backend/experiments/run_experiments.py first"), 404
    return app.response_class(path.read_text(encoding="utf-8"), mimetype="application/json")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
