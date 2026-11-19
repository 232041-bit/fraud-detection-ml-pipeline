"""Tests for Novelty 1 (drift-aware threshold) and Novelty 2 (uncertainty-aware decisions)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from novelties import (AdaptiveThresholdPolicy, DriftMonitor, MCDropoutMLP,  # noqa: E402
                       ThreeWayPolicy, population_stability_index, predictive_equality,
                       uncertainty_metrics)
from app import app, FEATURES  # noqa: E402

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ Novelty 1: drift
def test_psi_zero_for_identical_and_large_for_shift():
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) < 1e-9
    assert population_stability_index([0.5, 0.5], [0.9, 0.1]) > 0.25


def _frame(shift=0.0, n=4000):
    return pd.DataFrame({"x": RNG.normal(shift, 1, n), "y": RNG.normal(0, 1, n),
                         "c": RNG.choice(["a", "b"], n), "month": np.zeros(n)})


def test_drift_monitor_detects_shift_and_ignores_month():
    ref = _frame()
    mon = DriftMonitor(min_drifted_features=1).fit(ref, scores=RNG.random(4000))
    same = mon.check(_frame().assign(month=7), scores=RNG.random(4000))
    assert not same["drift_detected"]
    assert "month" not in same["features"]
    moved = mon.check(_frame(shift=1.0), scores=RNG.random(4000) ** 3)
    assert moved["drift_detected"] and moved["drifted_features"][0] == "x"
    assert moved["score_drift"]


def test_drift_monitor_roundtrip():
    mon = DriftMonitor().fit(_frame(), scores=RNG.random(4000))
    again = DriftMonitor.from_dict(json.loads(json.dumps(mon.to_dict())))
    rep_a = mon.check(_frame(0.5), RNG.random(500))
    rep_b = again.check(_frame(0.5), RNG.random(500))
    assert abs(rep_a["features"]["x"]["psi"] - rep_b["features"]["x"]["psi"]) < 0.05


def _scores(n, shift):
    y = (RNG.random(n) < 0.02).astype(int)
    s = RNG.normal(0, 1, n) + 2.0 * y + shift
    g = (RNG.random(n) < 0.3).astype(int)
    s = s + 0.5 * g                                   # group 1 gets higher scores (unfair)
    return s, g, y


def test_adaptive_threshold_restores_target_fpr_under_drift():
    s0, g0, y0 = _scores(20000, 0.0)
    thr = float(np.quantile(s0[y0 == 0], 0.95))
    static = AdaptiveThresholdPolicy(thr, mode="static")
    adaptive = AdaptiveThresholdPolicy(thr, mode="drift")
    s1, g1, y1 = _scores(20000, 0.8)                  # scores drift upwards
    for pol in (static, adaptive):
        pol.observe_scores(s1, g1)
        fb = pol.observe_labels(s1, g1, y1)
        pol.update({"drift_detected": True}, fb)
    s2, g2, y2 = _scores(20000, 0.8)
    fpr = lambda p: p.decide(s2, g2)[y2 == 0].mean()
    assert fpr(static) > 0.10                         # fixed threshold breaks the constraint
    assert abs(fpr(adaptive) - 0.05) < 0.01           # drift-aware threshold restores it


def test_fair_mode_improves_predictive_equality():
    s, g, y = _scores(40000, 0.0)
    thr = float(np.quantile(s[y == 0], 0.95))
    plain = AdaptiveThresholdPolicy(thr, mode="always")
    fair = AdaptiveThresholdPolicy(thr, mode="always", fair=True)
    for pol in (plain, fair):
        pol.observe_scores(s, g)
        pol.observe_labels(s, g, y)
        pol.update()
    pe_plain = predictive_equality(plain.decide(s, g), y, g)
    pe_fair = predictive_equality(fair.decide(s, g), y, g)
    assert pe_fair > 0.85 and pe_fair > pe_plain


# ------------------------------------------------------------------ Novelty 2: uncertainty
def test_uncertainty_metrics_decomposition():
    agree = uncertainty_metrics(np.array([[0.5, 0.9], [0.5, 0.9]]))
    disagree = uncertainty_metrics(np.array([[0.0, 0.9], [1.0, 0.9]]))
    assert agree["mutual_information"][0] < 1e-6        # same answer -> no epistemic part
    assert disagree["mutual_information"][0] > 0.9      # members disagree -> epistemic
    assert np.all(agree["predictive_entropy"] <= 1.0 + 1e-9)


def test_three_way_policy_respects_fpr_and_refers_uncertain():
    n = 20000
    y = (RNG.random(n) < 0.05).astype(int)
    p = np.clip(RNG.normal(0.3 + 0.4 * y, 0.15), 0, 1)
    met = {"mean": p, "mutual_information": RNG.random(n)}
    pol = ThreeWayPolicy(0.05, band_budget=0.03, epistemic_budget=0.02).fit(met, y)
    codes = pol.decide(met)
    assert ((codes == 2) & (y == 0)).sum() / (y == 0).sum() <= 0.05 + 1e-9
    assert 0.03 <= (codes == 1).mean() <= 0.08
    again = ThreeWayPolicy.from_params(pol.params())
    assert np.array_equal(again.decide(met), codes)


def test_mc_dropout_learns_and_is_stochastic():
    X = RNG.normal(size=(3000, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    net = MCDropoutMLP(hidden=(16,), epochs=8, lr=1e-2, dropout=0.2, seed=1).fit(X, y)
    acc = ((net.predict_proba(X)[:, 1] >= 0.5) == y).mean()
    assert acc > 0.9
    mc = net.member_probs(X[:50], T=10)
    assert mc.shape == (10, 50) and mc.std(axis=0).mean() > 0


# ------------------------------------------------------------------ API
CLIENT = app.test_client()


def test_predict_returns_three_way_decision_for_all_models():
    for model in ("random_forest", "logistic_regression", "deep_ensemble", "mc_dropout"):
        res = CLIENT.post(f"/predict?model={model}", json=PAYLOAD)
        body = res.get_json()
        assert res.status_code == 200, body
        assert body["decision"] in ("APPROVE", "MANUAL_REVIEW", "FLAG_FRAUD")
        assert 0 <= body["fraud_probability"] <= 1
        assert {"predictive_entropy", "mutual_information", "std"} <= set(body["uncertainty"])
        assert body["drift"]["adaptive_threshold"] == body["threshold"]


def test_fair_mode_uses_group_threshold():
    body = CLIENT.post("/predict?model=random_forest&fair=1", json=PAYLOAD).get_json()
    assert body["drift"]["fair_mode"] is True


def test_drift_batch_flow():
    CLIENT.post("/drift/reset?model=random_forest")
    demo = CLIENT.get("/drift/demo-batch?month=7&n=400").get_json()
    assert len(demo["records"]) == 400 and set(demo["records"][0]) == set(FEATURES)
    res = CLIENT.post("/drift/batch?model=random_forest",
                      json={"records": demo["records"], "labels": demo["labels"], "month": 7})
    body = res.get_json()
    assert res.status_code == 200, body
    assert "drift_detected" in body["report"] and body["report"]["top_features"]
    assert sum(body["update"]["decisions"].values()) == 400
    status = CLIENT.get("/drift/status?model=random_forest").get_json()
    assert len(status["log"]) == 1
    CLIENT.post("/drift/reset?model=random_forest")


def test_drift_batch_rejects_bad_input():
    assert CLIENT.post("/drift/batch", json={"records": []}).status_code == 400
    assert CLIENT.post("/drift/batch", json={"records": [PAYLOAD], "labels": [3]}).status_code == 400
    assert CLIENT.post("/predict?model=nope", json=PAYLOAD).status_code == 400


def test_novelty_summary_endpoint():
    res = CLIENT.get("/novelty/summary")
    assert res.status_code == 200
    assert {"drift", "uncertainty", "combined"} <= set(res.get_json())
