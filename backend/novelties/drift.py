"""Novelty 1 - Drift-Aware Fraud Detection Module.

The baseline pipeline fixes ONE decision threshold (5% FPR measured once, offline).
Fraud patterns and applicant populations change over time (the BAF months show a
large shift in e.g. ``velocity_4w`` and in fraud prevalence), so a fixed threshold
silently drifts away from the 5%-FPR business constraint.

This module adds an *online* layer that works with any fitted scorer:

1. ``DriftMonitor``            - monitors incoming micro-batches against a stored
                                  reference: Population Stability Index (PSI) and the
                                  two-sample Kolmogorov-Smirnov (KS) test on every input
                                  feature *and* on the model score distribution.
2. ``AdaptiveThresholdPolicy`` - when drift is detected (or performance feedback shows
                                  the realised FPR has left the tolerance band) the fraud
                                  threshold is recalibrated from (a) the most recent
                                  unlabeled score distribution and (b) delayed label
                                  feedback, so the realised FPR is pulled back to 5%.
                                  The optional *fair* mode recalibrates one threshold per
                                  age group (<50 / >=50) so that each group's FPR is 5%,
                                  i.e. predictive equality is maintained by construction.

Everything is plain numpy / scipy and JSON-serialisable so it can be stored inside
the model bundle and served by Flask.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

EPS = 1e-4
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25        # industry rule-of-thumb bands
KS_STAT_ALERT, KS_P_ALERT = 0.10, 0.01      # KS: effect size AND significance


# ----------------------------------------------------------------------------- helpers
def age_group(age: Any) -> np.ndarray:
    """Protected attribute used by the BAF fairness protocol: customer_age >= 50."""
    return (np.asarray(age, dtype=float) >= 50).astype(int)


def predictive_equality(pred: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """min(FPR_g) / max(FPR_g) over the two age groups (1.0 = perfectly fair)."""
    fprs = []
    for g in (0, 1):
        neg = (y == 0) & (group == g)
        if neg.sum():
            fprs.append(float(pred[neg].mean()))
    if len(fprs) < 2 or max(fprs) == 0:
        return 1.0
    return float(min(fprs) / max(fprs))


def _bin_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(float)
    return counts / max(counts.sum(), 1.0)


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((a - e) * ln(a / e)) over bins, with a small floor to avoid log(0)."""
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_severity(psi: float) -> str:
    return "major" if psi >= PSI_MAJOR else "moderate" if psi >= PSI_MODERATE else "stable"


# ----------------------------------------------------------------------------- monitor
@dataclass
class _FeatureRef:
    kind: str                       # "numeric" | "categorical"
    edges: list | None = None       # numeric bin edges
    levels: list | None = None      # categorical levels
    expected: list = field(default_factory=list)
    sample: list | None = None      # reference sample for the KS test


class DriftMonitor:
    """Reference distribution + PSI/KS drift tests for features and model scores."""

    def __init__(self, n_bins: int = 10, ks_sample: int = 5000, seed: int = 0,
                 min_drifted_features: int = 3, exclude: tuple = ("month",)):
        self.n_bins, self.ks_sample, self.seed = n_bins, ks_sample, seed
        self.exclude = tuple(exclude)          # time index columns are not "drift"
        self.min_drifted_features = min_drifted_features
        self.features: dict[str, _FeatureRef] = {}
        self.score_ref: _FeatureRef | None = None

    # -- fitting ------------------------------------------------------------------
    def _ref_numeric(self, values: np.ndarray, rng) -> _FeatureRef:
        values = values[np.isfinite(values)]
        edges = _bin_edges(values, self.n_bins)
        sample = rng.choice(values, size=min(self.ks_sample, len(values)), replace=False)
        return _FeatureRef("numeric", edges.tolist(), None,
                           _proportions(values, edges).tolist(), sample.tolist())

    def fit(self, frame, scores: np.ndarray | None = None) -> "DriftMonitor":
        rng = np.random.default_rng(self.seed)
        for col in frame.columns:
            if col in self.exclude:
                continue
            s = frame[col]
            if s.dtype == object or str(s.dtype).startswith(("category", "str", "string")) \
                    or s.nunique() <= 12:
                levels = sorted(map(str, s.astype(str).unique()))
                freq = s.astype(str).value_counts(normalize=True)
                self.features[col] = _FeatureRef(
                    "categorical", None, levels, [float(freq.get(lv, 0.0)) for lv in levels])
            else:
                self.features[col] = self._ref_numeric(s.to_numpy(float), rng)
        if scores is not None:
            self.score_ref = self._ref_numeric(np.asarray(scores, float), rng)
        return self

    def rebase(self, frame, scores: np.ndarray | None = None) -> "DriftMonitor":
        """Re-anchor the reference on the data a recalibration was based on, so later
        alerts measure *new* change since the last adaptation (not the same old shift)."""
        fresh = DriftMonitor(self.n_bins, self.ks_sample, self.seed,
                             self.min_drifted_features, self.exclude).fit(frame, scores)
        self.features, self.score_ref = fresh.features, fresh.score_ref
        return self

    # -- testing ------------------------------------------------------------------
    @staticmethod
    def _test(ref: _FeatureRef, values) -> dict:
        if ref.kind == "categorical":
            vals = np.asarray(values).astype(str)
            actual = [float(np.mean(vals == lv)) for lv in ref.levels]
            unseen = float(np.mean(~np.isin(vals, ref.levels)))
            psi = population_stability_index(ref.expected + [EPS], actual + [unseen])
            return {"psi": psi, "ks_stat": None, "ks_p": None, "severity": psi_severity(psi)}
        vals = np.asarray(values, float)
        vals = vals[np.isfinite(vals)]
        psi = population_stability_index(ref.expected, _proportions(vals, np.asarray(ref.edges)))
        ks = ks_2samp(np.asarray(ref.sample), vals)
        return {"psi": psi, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                "severity": psi_severity(psi)}

    def check(self, frame, scores: np.ndarray | None = None) -> dict:
        """Compare an incoming batch with the reference and return a drift report."""
        feats = {c: self._test(r, frame[c].to_numpy()) for c, r in self.features.items()
                 if c in frame.columns}
        drifted = sorted(
            [c for c, r in feats.items() if r["psi"] >= PSI_MODERATE or
             (r["ks_stat"] is not None and r["ks_stat"] >= KS_STAT_ALERT and r["ks_p"] < KS_P_ALERT)],
            key=lambda c: -feats[c]["psi"])
        score = None
        if scores is not None and self.score_ref is not None:
            score = self._test(self.score_ref, scores)
        score_drift = bool(score and (score["psi"] >= PSI_MODERATE or
                                      (score["ks_stat"] >= KS_STAT_ALERT and score["ks_p"] < KS_P_ALERT)))
        feature_drift = len(drifted) >= self.min_drifted_features or any(
            feats[c]["psi"] >= PSI_MAJOR for c in drifted)
        max_psi = max([r["psi"] for r in feats.values()] + [score["psi"] if score else 0.0])
        return {"n": int(len(frame)), "features": feats, "drifted_features": drifted,
                "score": score, "feature_drift": bool(feature_drift),
                "score_drift": score_drift,
                "drift_detected": bool(feature_drift or score_drift),
                "severity": psi_severity(max_psi), "max_psi": float(max_psi)}

    # -- (de)serialisation ----------------------------------------------------------
    def to_dict(self) -> dict:
        enc = lambda r: None if r is None else {k: (None if v is None else
                                                     [None if (isinstance(x, float) and not np.isfinite(x)) else x
                                                      for x in v] if isinstance(v, list) else v)
                                                 for k, v in r.__dict__.items()}
        return {"n_bins": self.n_bins, "ks_sample": self.ks_sample, "seed": self.seed,
                "min_drifted_features": self.min_drifted_features,
                "exclude": list(self.exclude),
                "features": {c: enc(r) for c, r in self.features.items()},
                "score_ref": enc(self.score_ref)}

    @classmethod
    def from_dict(cls, d: dict) -> "DriftMonitor":
        m = cls(d["n_bins"], d["ks_sample"], d["seed"], d.get("min_drifted_features", 3),
                tuple(d.get("exclude", ("month",))))

        def dec(r):
            if r is None:
                return None
            r = dict(r)
            if r.get("edges") is not None:
                e = list(r["edges"])
                e[0], e[-1] = -np.inf, np.inf
                r["edges"] = e
            return _FeatureRef(**r)
        m.features = {c: dec(r) for c, r in d["features"].items()}
        m.score_ref = dec(d["score_ref"])
        return m


# ----------------------------------------------------------------------------- policy
class AdaptiveThresholdPolicy:
    """Drift-triggered, fairness-aware recalibration of the fraud decision threshold.

    Information available when the threshold for batch *b* is chosen:
      * unlabeled model scores of earlier batches (immediately),
      * fraud labels of a batch only ``label_delay`` batches later (chargeback delay).

    Recalibration (hybrid quantile estimator).  With prevalence pi and recall r the share
    of ALL traffic above a 5%-FPR threshold is  a = target*(1-pi) + r*pi.  pi and r are
    estimated from the most recent *labeled* batches, ``a`` is then applied to the most
    recent *unlabeled* score distribution, which already reflects the drift.  When enough
    labeled negatives exist the label-based quantile of negatives is blended in.
    In ``fair`` mode the procedure runs separately for each age group.
    """

    def __init__(self, base_threshold: float, target_fpr: float = 0.05,
                 fair: bool = False, mode: str = "drift", window: int = 2,
                 fpr_tolerance: float = 0.01, min_negatives: int = 2000,
                 group_thresholds: dict | None = None):
        assert mode in ("static", "drift", "always")
        self.target, self.fair, self.mode = target_fpr, fair, mode
        self.window, self.tol, self.min_neg = window, fpr_tolerance, min_negatives
        self.threshold = float(base_threshold)
        self.base_threshold = float(base_threshold)
        self.group_thresholds = {int(k): float(v) for k, v in (group_thresholds or
                                 {0: base_threshold, 1: base_threshold}).items()}
        self.unlabeled: deque = deque(maxlen=window)    # (scores, groups)
        self.labeled: deque = deque(maxlen=window)      # (scores, groups, y)
        self.history: list[dict] = []

    # -- feedback -------------------------------------------------------------------
    def observe_scores(self, scores, groups) -> None:
        self.unlabeled.append((np.asarray(scores, float), np.asarray(groups, int)))

    def observe_labels(self, scores, groups, y) -> dict:
        """Delayed label feedback; returns realised performance of the current policy."""
        s, g, y = np.asarray(scores, float), np.asarray(groups, int), np.asarray(y, int)
        self.labeled.append((s, g, y))
        pred = self.decide(s, g)
        neg = y == 0
        return {"fpr": float(pred[neg].mean()) if neg.any() else 0.0,
                "recall": float(pred[~neg].mean()) if (~neg).any() else 0.0}

    # -- decision -------------------------------------------------------------------
    def threshold_for(self, groups) -> np.ndarray:
        g = np.asarray(groups, int)
        if not self.fair:
            return np.full(g.shape, self.threshold)
        return np.where(g == 1, self.group_thresholds[1], self.group_thresholds[0])

    def decide(self, scores, groups) -> np.ndarray:
        return np.asarray(scores, float) >= self.threshold_for(groups)

    # -- recalibration --------------------------------------------------------------
    def _estimate(self, mask_fn) -> float | None:
        lab = [(s[mask_fn(g)], y[mask_fn(g)]) for s, g, y in self.labeled]
        unl = [s[mask_fn(g)] for s, g in self.unlabeled]
        if not lab or not unl:
            return None
        s_lab = np.concatenate([s for s, _ in lab])
        y_lab = np.concatenate([y for _, y in lab])
        s_now = np.concatenate(unl)
        neg = s_lab[y_lab == 0]
        if len(neg) < 50 or len(s_now) < 50:
            return None
        t_lab = float(np.quantile(neg, 1 - self.target))        # 5% FPR on labeled data
        pi = float(y_lab.mean())
        r = float((s_lab[y_lab == 1] >= t_lab).mean()) if (y_lab == 1).any() else 0.5
        alert_share = self.target * (1 - pi) + r * pi
        t_unl = float(np.quantile(s_now, 1 - alert_share))      # drift-aware, label free
        w = min(1.0, len(neg) / (4.0 * self.min_neg))           # trust labels when plentiful
        return float((1 - 0.5 * w) * t_unl + 0.5 * w * t_lab)

    def update(self, drift_report: dict | None = None, feedback: dict | None = None) -> dict:
        """Decide whether to recalibrate before scoring the next batch."""
        reason = None
        if self.mode == "always":
            reason = "scheduled"
        elif self.mode == "drift":
            if drift_report and drift_report.get("drift_detected"):
                reason = "data/score drift (PSI/KS)"
            if feedback and abs(feedback["fpr"] - self.target) > self.tol:
                reason = (reason + " + " if reason else "") + "FPR outside tolerance"
        old = (self.threshold, dict(self.group_thresholds))
        if reason:
            t = self._estimate(lambda g: np.ones(len(g), bool))
            if t is not None:
                self.threshold = t
            if self.fair:
                for k in (0, 1):
                    tg = self._estimate(lambda g, k=k: g == k)
                    if tg is not None:
                        self.group_thresholds[k] = tg
        rec = {"recalibrated": bool(reason) and (old[0] != self.threshold or
                                                old[1] != self.group_thresholds),
               "reason": reason, "threshold": self.threshold,
               "group_thresholds": dict(self.group_thresholds),
               "previous_threshold": old[0]}
        self.history.append(rec)
        return rec

    def state(self) -> dict:
        return {"mode": self.mode, "fair": self.fair, "target_fpr": self.target,
                "threshold": self.threshold, "base_threshold": self.base_threshold,
                "group_thresholds": {str(k): v for k, v in self.group_thresholds.items()},
                "labeled_batches": len(self.labeled), "unlabeled_batches": len(self.unlabeled),
                "recalibrations": int(sum(h["recalibrated"] for h in self.history))}
