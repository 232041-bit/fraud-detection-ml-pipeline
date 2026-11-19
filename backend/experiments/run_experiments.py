r"""Reproduce the evaluation of both novelties and write docs/novelty_results.json
(+ PNG figures in docs/figures/ when matplotlib is installed).

    python backend/experiments/run_experiments.py [--data FraudDetection_Dataset.csv]

Experiment 1 - Drift-aware threshold adaptation (Novelty 1)
    A HistGradientBoosting scorer is trained on months 0-1, its 5%-FPR threshold is set on
    month 2, then months 3-7 arrive as a stream of micro-batches (4 per month). Fraud labels
    arrive with a delay of one batch. Policies compared:
        static        - the fixed threshold of the baseline paper/pipeline
        drift         - PSI/KS-triggered recalibration (Novelty 1)
        drift_fair    - drift + per-age-group recalibration (fairness-aware)
        always        - ablation: recalibrate every batch regardless of drift
    Experiment 1b repeats the stream with the deployed Random Forest on months 6 -> 7.

Experiment 2 - Uncertainty-aware three-way decisions (Novelty 2)
    Uncertainty models / policies from backend/train_novelties.py (calibrated on month 6)
    are evaluated on the untouched month 7 against the binary 5%-FPR baseline.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from novelties import (AdaptiveThresholdPolicy, DriftMonitor, ThreeWayPolicy,  # noqa: E402
                       TreeEnsembleUncertainty, age_group, predictive_equality,
                       uncertainty_metrics)
from novelties.uncertainty import evaluate_three_way  # noqa: E402
from train_novelties import CATEGORICAL, TARGET, find_data  # noqa: E402

warnings.filterwarnings("ignore")
TARGET_FPR = 0.05


# =============================================================================== drift
def run_stream(frames, scores, groups, ys, monitor, policy, label_delay=1, rebase=True):
    """Online simulation. Batch b is decided with information from batches < b only."""
    import copy
    monitor = copy.deepcopy(monitor) if rebase else monitor
    rows, pending, last_report, feedback = [], [], None, None
    for b in range(len(frames)):
        rec = policy.update(last_report, feedback)
        feedback = None
        if rebase and rec["recalibrated"] and b > 0:
            monitor.rebase(frames[b - 1], scores[b - 1])
        pred = policy.decide(scores[b], groups[b])
        y, g = ys[b], groups[b]
        neg = y == 0
        rows.append({"batch": b, "fpr": float(pred[neg].mean()),
                     "recall": float(pred[~neg].mean()),
                     "pe_age": predictive_equality(pred, y, g),
                     "threshold": float(policy.threshold),
                     "group_thresholds": {str(k): float(v) for k, v in policy.group_thresholds.items()},
                     "recalibrated": bool(rec["recalibrated"]), "reason": rec["reason"],
                     "n": int(len(y)), "fraud_rate": float(y.mean())})
        # monitoring of the batch that just completed (features + scores, no labels)
        last_report = monitor.check(frames[b], scores[b])
        rows[-1].update({"drift_detected": last_report["drift_detected"],
                         "max_psi": last_report["max_psi"],
                         "score_psi": last_report["score"]["psi"] if last_report["score"] else None,
                         "n_drifted_features": len(last_report["drifted_features"]),
                         "top_drifted": last_report["drifted_features"][:5]})
        policy.observe_scores(scores[b], groups[b])
        pending.append(b)
        while pending and b - pending[0] >= label_delay:
            k = pending.pop(0)
            feedback = policy.observe_labels(scores[k], groups[k], ys[k])
    return rows


def summarise(rows, months):
    df = pd.DataFrame(rows)
    df["month"] = months
    dev = (df.fpr - TARGET_FPR).abs()
    per_month = df.groupby("month").apply(lambda d: pd.Series({
        "fpr": np.average(d.fpr, weights=d.n), "recall": np.average(d.recall, weights=d.n),
        "pe_age": d.pe_age.mean()})).reset_index().to_dict("records")
    return {"mean_abs_fpr_dev": float(dev.mean()), "max_fpr": float(df.fpr.max()),
            "mean_fpr": float(df.fpr.mean()), "mean_recall": float(df.recall.mean()),
            "mean_pe_age": float(df.pe_age.mean()),
            "batches_within_tolerance": float((dev <= 0.01).mean()),
            "recalibrations": int(df.recalibrated.sum()), "per_month": per_month,
            "batches": rows}


def make_batches(df, months, per_month, seed=0):
    out = []
    for m in months:
        part = df[df.month == m].sample(frac=1.0, random_state=seed + m).reset_index(drop=True)
        for chunk in np.array_split(np.arange(len(part)), per_month):
            out.append((m, part.iloc[chunk]))
    return out


def experiment_drift(df, features, numeric):
    print("\n=== Experiment 1: drift-aware threshold (HGB, months 0-1 -> stream 3-7) ===")
    enc = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1), CATEGORICAL)],
                            remainder="passthrough")
    train, ref = df[df.month <= 1], df[df.month == 2]
    enc.fit(train[features])
    n_cat = len(CATEGORICAL)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                           categorical_features=list(range(n_cat)),
                                           random_state=0)
    model.fit(enc.transform(train[features]), train[TARGET])
    score = lambda fr: model.predict_proba(enc.transform(fr[features]))[:, 1]
    s_ref, y_ref, g_ref = score(ref), ref[TARGET].to_numpy(), age_group(ref.customer_age)
    base_thr = float(np.quantile(s_ref[y_ref == 0], 1 - TARGET_FPR))
    grp_thr = {g: float(np.quantile(s_ref[(y_ref == 0) & (g_ref == g)], 1 - TARGET_FPR))
               for g in (0, 1)}
    print(f"reference month 2: AUC={roc_auc_score(y_ref, s_ref):.4f} threshold={base_thr:.4f}")

    batches = make_batches(df, range(3, 8), per_month=4)
    frames = [b[features] for _, b in batches]
    scores = [score(b) for _, b in batches]
    groups = [age_group(b.customer_age) for _, b in batches]
    ys = [b[TARGET].to_numpy() for _, b in batches]
    months = [m for m, _ in batches]
    monitor = DriftMonitor(seed=0).fit(ref[features], scores=s_ref)

    results = {}
    configs = {"static": dict(mode="static"), "drift": dict(mode="drift"),
               "drift_fair": dict(mode="drift", fair=True), "always": dict(mode="always")}
    for name, cfg in configs.items():
        pol = AdaptiveThresholdPolicy(base_thr, TARGET_FPR, group_thresholds=grp_thr, **cfg)
        results[name] = summarise(run_stream(frames, scores, groups, ys, monitor, pol), months)
        r = results[name]
        print(f"{name:11s} |FPR-5%|={r['mean_abs_fpr_dev']:.4f} maxFPR={r['max_fpr']:.4f} "
              f"recall={r['mean_recall']:.4f} PE(age)={r['mean_pe_age']:.3f} "
              f"in-tol={r['batches_within_tolerance']:.2f} recal={r['recalibrations']}")
    # feature-level drift per month (for the drift heat-map)
    psi_by_month = {}
    for m in range(3, 8):
        part = df[df.month == m]
        rep = monitor.check(part[features], score(part))
        psi_by_month[m] = {"features": {c: v["psi"] for c, v in rep["features"].items()},
                           "score_psi": rep["score"]["psi"], "drift": rep["drift_detected"],
                           "auc": float(roc_auc_score(part[TARGET], score(part)))}
    return {"setup": {"model": "HistGradientBoosting", "train_months": [0, 1],
                      "reference_month": 2, "stream_months": [3, 7], "batches_per_month": 4,
                      "label_delay_batches": 1, "base_threshold": base_thr,
                      "base_group_thresholds": grp_thr},
            "policies": results, "psi_by_month": psi_by_month}


def experiment_drift_rf(df, features, base, nov):
    print("\n=== Experiment 1b: deployed Random Forest, calibration month 6 -> stream month 7 ===")
    rf = base["pipelines"]["random_forest"]
    batches = make_batches(df, [7], per_month=8)
    frames = [b[features] for _, b in batches]
    scores = [rf.predict_proba(b[features])[:, 1] for _, b in batches]
    groups = [age_group(b.customer_age) for _, b in batches]
    ys = [b[TARGET].to_numpy() for _, b in batches]
    monitor = DriftMonitor.from_dict(nov["drift_reference"]["random_forest"])
    gthr = nov["group_thresholds"]["random_forest"]
    results = {}
    for name, cfg in {"static": dict(mode="static"), "drift": dict(mode="drift"),
                      "drift_fair": dict(mode="drift", fair=True)}.items():
        pol = AdaptiveThresholdPolicy(nov["thresholds"]["random_forest"], TARGET_FPR,
                                      group_thresholds=gthr, **cfg)
        results[name] = summarise(run_stream(frames, scores, groups, ys, monitor, pol),
                                  [7] * len(frames))
        r = results[name]
        print(f"{name:11s} |FPR-5%|={r['mean_abs_fpr_dev']:.4f} recall={r['mean_recall']:.4f} "
              f"PE(age)={r['mean_pe_age']:.3f}")
    return {"setup": {"model": "RandomForest (deployed)", "calibration_month": 6,
                      "stream_month": 7, "batches": 8, "label_delay_batches": 1},
            "policies": results}


# =============================================================================== uncertainty
def experiment_uncertainty(df, features, base, nov):
    print("\n=== Experiment 2: uncertainty-aware three-way decisions (cal 6 -> test 7) ===")
    cal, test = df[df.month == 6], df[df.month == 7]
    y_cal, y = cal[TARGET].to_numpy(), test[TARGET].to_numpy()
    ests = {"random_forest": TreeEnsembleUncertainty(base["pipelines"]["random_forest"]),
            "deep_ensemble": nov["uncertainty_models"]["deep_ensemble"],
            "mc_dropout": nov["uncertainty_models"]["mc_dropout"]}
    out = {}
    for name, est in ests.items():
        m_cal = uncertainty_metrics(est.member_probs(cal[features]))
        m_te = uncertainty_metrics(est.member_probs(test[features]))
        pol = ThreeWayPolicy.from_params(nov["policies"][name])
        binary = (m_te["mean"] >= pol.t_fraud).astype(int) * 2          # 0 / 2 only
        res = {"auc": float(roc_auc_score(y, m_te["mean"])),
               "binary_baseline": evaluate_three_way(binary, y),
               "three_way": evaluate_three_way(pol.decide(m_te), y), "ablations": {},
               "budget_sweep": []}
        for key in ("mutual_information", "predictive_entropy", "std"):
            for label, band, epi in (("grey_zone_only", 0.03, 0.0),
                                     ("uncertainty_only", 0.0, 0.03),
                                     ("grey_zone+uncertainty", 0.03, 0.02)):
                p = ThreeWayPolicy(TARGET_FPR, band, epi, key).fit(m_cal, y_cal)
                res["ablations"][f"{key}|{label}"] = evaluate_three_way(p.decide(m_te), y)
        for budget in (0.0, 0.01, 0.02, 0.05, 0.08, 0.12):
            p = ThreeWayPolicy(TARGET_FPR, budget * 0.6, budget * 0.4).fit(m_cal, y_cal)
            r = evaluate_three_way(p.decide(m_te), y)
            res["budget_sweep"].append({"budget": budget, **r})
        # do uncertain cases really contain more errors?  error rate per uncertainty decile
        binpred = m_te["mean"] >= pol.t_fraud
        err = (binpred != (y == 1)).astype(float)
        dec = pd.qcut(pd.Series(m_te["mutual_information"]).rank(method="first"), 10, labels=False)
        res["error_by_uncertainty_decile"] = [float(err[dec == d].mean()) for d in range(10)]
        out[name] = res
        b, t = res["binary_baseline"], res["three_way"]
        print(f"{name:14s} AUC={res['auc']:.4f} | binary: auto-err={b['auto_error_rate']:.4f} "
              f"FPR={b['auto_fpr']:.4f} recall={b['auto_recall']:.4f} | 3-way: auto-err="
              f"{t['auto_error_rate']:.4f} review={t['review_rate']:.4f} autoFPR={t['auto_fpr']:.4f}"
              f" recall+review={t['recall_with_review']:.4f} errors {b['auto_errors']}->{t['auto_errors']}")
    return out


def experiment_combined(df, features, base, nov):
    print("\n=== Experiment 3: Novelty 1 + 2 combined (month 7 stream, drift-adapted 3-way) ===")
    test = df[df.month == 7].sample(frac=1.0, random_state=7).reset_index(drop=True)
    chunks = np.array_split(np.arange(len(test)), 8)
    ests = {"random_forest": TreeEnsembleUncertainty(base["pipelines"]["random_forest"]),
            "deep_ensemble": nov["uncertainty_models"]["deep_ensemble"],
            "mc_dropout": nov["uncertainty_models"]["mc_dropout"]}
    out = {}
    for name, est in ests.items():
        met_all = uncertainty_metrics(est.member_probs(test[features]))
        y_all, g_all = test[TARGET].to_numpy(), age_group(test.customer_age)
        three = ThreeWayPolicy.from_params(nov["policies"][name])
        monitor = DriftMonitor.from_dict(nov["drift_reference"][name])
        res = {}
        for mode in ("static", "drift"):
            pol = AdaptiveThresholdPolicy(nov["thresholds"][name], TARGET_FPR, mode=mode)
            codes, report, fb = [], None, None
            for b, idx in enumerate(chunks):
                pol.update(report, fb)
                met = {k: v[idx] for k, v in met_all.items()}
                codes.append(three.decide(met, t_fraud=pol.threshold))
                report = monitor.check(test.loc[idx, features], met["mean"])
                pol.observe_scores(met["mean"], g_all[idx])
                if b >= 1:
                    prev = chunks[b - 1]
                    fb = pol.observe_labels(met_all["mean"][prev], g_all[prev], y_all[prev])
            res[mode] = evaluate_three_way(np.concatenate(codes), y_all[np.concatenate(chunks)])
        out[name] = res
        for mode, r in res.items():
            print(f"{name:14s} {mode:7s} autoFPR={r['auto_fpr']:.4f} auto-err={r['auto_error_rate']:.4f}"
                  f" review={r['review_rate']:.4f} recall+review={r['recall_with_review']:.4f}")
    return out


# =============================================================================== figures
def figures(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figures")
        return
    fig_dir = ROOT / "docs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = results["drift"]["policies"]
    colors = {"static": "#888888", "drift": "#2a6fdb", "drift_fair": "#1b9e77", "always": "#d95f02"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, r in d.items():
        fpr = [b["fpr"] for b in r["batches"]]
        pe = [b["pe_age"] for b in r["batches"]]
        ax[0].plot(fpr, label=name, color=colors[name], marker="o", ms=3)
        ax[1].plot(pe, label=name, color=colors[name], marker="o", ms=3)
    ax[0].axhline(TARGET_FPR, ls="--", c="k", lw=0.8)
    ax[0].set(title="Realised FPR per micro-batch (months 3-7)", xlabel="batch", ylabel="FPR")
    ax[1].set(title="Predictive equality (age)", xlabel="batch", ylabel="min/max FPR ratio")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "drift_fpr_stream.png", dpi=150)
    plt.close(fig)

    u = results["uncertainty"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, r in u.items():
        sw = r["budget_sweep"]
        ax[0].plot([s["review_rate"] for s in sw], [s["auto_error_rate"] for s in sw],
                   marker="o", label=name)
        ax[1].plot(range(1, 11), r["error_by_uncertainty_decile"], marker="o", label=name)
    ax[0].set(title="Automated error rate vs review rate", xlabel="share sent to manual review",
              ylabel="error rate of automated decisions")
    ax[1].set(title="Error rate by epistemic-uncertainty decile", xlabel="decile (1 = most certain)",
              ylabel="binary decision error rate")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "uncertainty_tradeoff.png", dpi=150)
    plt.close(fig)
    print("figures written to docs/figures/")


def compact_summary(results):
    """Small JSON consumed by the Angular dashboard (also works on static GitHub Pages)."""
    keep = ("mean_abs_fpr_dev", "max_fpr", "mean_fpr", "mean_recall", "mean_pe_age",
            "batches_within_tolerance", "recalibrations")
    pol = lambda r: {**{k: r[k] for k in keep},
                     "fpr": [round(b["fpr"], 5) for b in r["batches"]],
                     "recall": [round(b["recall"], 5) for b in r["batches"]],
                     "pe_age": [round(b["pe_age"], 4) for b in r["batches"]],
                     "threshold": [round(b["threshold"], 5) for b in r["batches"]],
                     "recalibrated": [b["recalibrated"] for b in r["batches"]],
                     "drift_detected": [b["drift_detected"] for b in r["batches"]]}
    d = results["drift"]
    months = [m for m in range(3, 8) for _ in range(d["setup"]["batches_per_month"])]
    top = sorted(d["psi_by_month"][7]["features"].items(), key=lambda kv: -kv[1])[:8]
    psi = {"features": [c for c, _ in top],
           "months": list(d["psi_by_month"].keys()),
           "values": [[round(d["psi_by_month"][m]["features"][c], 4) for c, _ in top]
                      for m in d["psi_by_month"]],
           "score_psi": [round(d["psi_by_month"][m]["score_psi"], 4) for m in d["psi_by_month"]],
           "auc": [round(d["psi_by_month"][m]["auc"], 4) for m in d["psi_by_month"]]}
    unc = {}
    for name, r in results["uncertainty"].items():
        unc[name] = {"auc": r["auc"], "binary": r["binary_baseline"], "three_way": r["three_way"],
                     "budget_sweep": [{k: s[k] for k in ("budget", "review_rate", "auto_error_rate",
                                                         "auto_fpr", "recall_with_review")}
                                      for s in r["budget_sweep"]],
                     "error_by_uncertainty_decile": r["error_by_uncertainty_decile"],
                     "ablations": {k: {m: v[m] for m in ("review_rate", "auto_error_rate",
                                                         "recall_with_review")}
                                   for k, v in r["ablations"].items()}}
    return {"target_fpr": TARGET_FPR,
            "drift": {"setup": d["setup"], "batch_months": months,
                      "policies": {k: pol(v) for k, v in d["policies"].items()}, "psi": psi},
            "drift_rf": {"setup": results["drift_rf"]["setup"],
                         "policies": {k: pol(v) for k, v in results["drift_rf"]["policies"].items()}},
            "uncertainty": unc, "combined": results["combined"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    args = ap.parse_args()
    t0 = time.time()
    base = joblib.load(ROOT / "backend/models/model_bundle.joblib")
    nov = joblib.load(ROOT / "backend/models/novelty_bundle.joblib")
    features = base["features"]
    numeric = [c for c in features if c not in CATEGORICAL]
    df = pd.read_csv(find_data(args.data))
    results = {"drift": experiment_drift(df, features, numeric),
               "drift_rf": experiment_drift_rf(df, features, base, nov),
               "uncertainty": experiment_uncertainty(df, features, base, nov),
               "combined": experiment_combined(df, features, base, nov)}
    (ROOT / "docs/novelty_results.json").write_text(json.dumps(results, indent=1, default=float))
    summary = compact_summary(results)
    txt = json.dumps(summary, indent=1, default=float)
    (ROOT / "docs/novelty_summary.json").write_text(txt)
    (ROOT / "frontend/public").mkdir(parents=True, exist_ok=True)
    (ROOT / "frontend/public/novelty_summary.json").write_text(txt)   # static dashboard data
    figures(results)
    print(f"done in {time.time() - t0:.0f}s -> docs/novelty_results.json")


if __name__ == "__main__":
    main()
