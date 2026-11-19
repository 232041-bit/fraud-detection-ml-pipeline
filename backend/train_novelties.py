r"""Train / calibrate the two research novelties and save backend/models/novelty_bundle.joblib.

Temporal protocol (same chronological idea as the paper / baseline):
    months 0-5  -> training of the new uncertainty models (deep ensemble, MC dropout)
                   and the *feature* reference distribution of the drift monitor
    month 6     -> calibration: 5%-FPR thresholds, three-way policy, score reference
    month 7     -> untouched test month (used by backend/experiments/*)

Run from the project root (needs the full BAF csv, ~160 MB, tracked with git LFS):
    python backend/train_novelties.py [--data FraudDetection_Dataset.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from novelties import (DeepEnsemble, DriftMonitor, MCDropoutMLP,  # noqa: E402
                       ThreeWayPolicy, TreeEnsembleUncertainty, age_group, uncertainty_metrics)

warnings.filterwarnings("ignore")
TARGET = "fraud_bool"
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
TARGET_FPR = 0.05


def find_data(arg: str | None) -> Path:
    for p in [arg, "data/FraudDetection_Dataset.csv", "FraudDetection_Dataset.csv"]:
        if p and (ROOT / p).exists() and (ROOT / p).stat().st_size > 10_000_000:
            return ROOT / p
        if p and Path(p).exists() and Path(p).stat().st_size > 10_000_000:
            return Path(p)
    sys.exit("Full BAF csv not found (run `git lfs pull` or pass --data PATH).")


def undersample(df: pd.DataFrame, ratio: int, seed: int) -> pd.DataFrame:
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0].sample(n=min(len(pos) * ratio, (df[TARGET] == 0).sum()),
                                     random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--ratio", type=int, default=10, help="negatives per fraud when bagging")
    args = ap.parse_args()

    t0 = time.time()
    base = joblib.load(ROOT / "backend/models/model_bundle.joblib")
    features = base["features"]
    numeric = [c for c in features if c not in CATEGORICAL]
    df = pd.read_csv(find_data(args.data))
    train, cal = df[df.month <= 5], df[df.month == 6].reset_index(drop=True)
    print(f"train={len(train):,} cal(month 6)={len(cal):,}")

    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), CATEGORICAL)])
    pre.fit(train[features])

    # ---- deep ensemble: bagged + differently initialised MLPs ----------------------------
    members = []
    for m in range(args.members):
        part = undersample(train, args.ratio, seed=100 + m)
        net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                            learning_rate_init=1e-3, max_iter=25, random_state=100 + m)
        net.fit(pre.transform(part[features]), part[TARGET].to_numpy())
        members.append(net)
        print(f"deep-ensemble member {m + 1}/{args.members} done ({time.time() - t0:.0f}s)")
    deep = DeepEnsemble(pre, members)

    # ---- MC dropout network ----------------------------------------------------------------
    part = undersample(train, args.ratio, seed=7)
    mcd = MCDropoutMLP(pre, hidden=(64, 32), dropout=0.2, epochs=15, mc_samples=30, seed=7)
    mcd.fit(pre.transform(part[features]), part[TARGET].to_numpy(), verbose=True)
    print(f"mc-dropout done ({time.time() - t0:.0f}s)")

    # ---- calibration on month 6 ------------------------------------------------------------
    estimators = {"random_forest": TreeEnsembleUncertainty(base["pipelines"]["random_forest"]),
                  "deep_ensemble": deep, "mc_dropout": mcd}
    y_cal, g_cal = cal[TARGET].to_numpy(), age_group(cal.customer_age)
    Xc = cal[features]
    policies, thresholds, group_thr, cal_scores = {}, {}, {}, {}
    for name, est in estimators.items():
        met = uncertainty_metrics(est.member_probs(Xc))
        policies[name] = ThreeWayPolicy(TARGET_FPR).fit(met, y_cal)
        cal_scores[name] = met["mean"]
    lr = base["pipelines"]["logistic_regression"]
    cal_scores["logistic_regression"] = lr.predict_proba(Xc)[:, 1]
    # LR has no stochastic members -> grey zone only, no epistemic referral
    policies["logistic_regression"] = ThreeWayPolicy(TARGET_FPR, epistemic_budget=0.0).fit(
        uncertainty_metrics(cal_scores["logistic_regression"][None, :]), y_cal)

    for name, s in cal_scores.items():
        neg = y_cal == 0
        thresholds[name] = float(np.quantile(s[neg], 1 - TARGET_FPR))
        group_thr[name] = {g: float(np.quantile(s[neg & (g_cal == g)], 1 - TARGET_FPR))
                           for g in (0, 1)}
        print(f"{name:20s} 5%-FPR threshold={thresholds[name]:.4f} groups={group_thr[name]}")

    # ---- drift monitor references ----------------------------------------------------------
    ref_rows = train[features].sample(n=min(200_000, len(train)), random_state=0)
    monitors = {name: DriftMonitor(seed=0).fit(ref_rows, scores=s).to_dict()
                for name, s in cal_scores.items()}

    bundle = {"version": "2.0-novelties", "target_fpr": TARGET_FPR,
              "uncertainty_models": {"deep_ensemble": deep, "mc_dropout": mcd},
              "policies": {k: v.params() for k, v in policies.items()},
              "thresholds": thresholds, "group_thresholds": group_thr,
              "drift_reference": monitors,
              "protocol": {"train_months": [0, 5], "calibration_month": 6, "test_month": 7}}
    out = ROOT / "backend/models/novelty_bundle.joblib"
    joblib.dump(bundle, out, compress=3)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/novelty_calibration.json").write_text(json.dumps(
        {"policies": bundle["policies"], "thresholds": thresholds,
         "group_thresholds": group_thr, "protocol": bundle["protocol"]}, indent=2))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
