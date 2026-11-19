"""Novelty 2 - Uncertainty-Aware Fraud Detection.

The baseline returns a hard Fraud / Legitimate label even when the model is unsure.
Here every scorer exposes a *set* of stochastic predictions (ensemble members or
Monte-Carlo dropout passes), from which we compute

* mean fraud probability                      p̄ = mean_m p_m
* predictive entropy (total uncertainty)       H[p̄]
* expected entropy (aleatoric part)            E_m H[p_m]
* mutual information (epistemic part, BALD)    MI = H[p̄] - E_m H[p_m]
* member standard deviation                    std_m p_m

and a calibrated three-way decision policy:

    APPROVE        low risk and confident
    FLAG_FRAUD     high risk (>= 5%-FPR threshold) and confident
    MANUAL_REVIEW  score in the grey zone OR epistemic uncertainty above budget

Three uncertainty estimators are provided:
  * ``TreeEnsembleUncertainty`` - disagreement of the trees of the deployed Random Forest
  * ``DeepEnsemble``            - M independently initialised / bagged MLPs
  * ``MCDropoutMLP``            - numpy MLP trained with dropout, dropout kept ON at
                                  inference (T stochastic forward passes, Gal & Ghahramani)
"""
from __future__ import annotations

import numpy as np

LOG2 = np.log(2.0)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / LOG2       # bits, max 1


def uncertainty_metrics(member_probs: np.ndarray) -> dict:
    """member_probs: array (M, n) of fraud probabilities from M stochastic predictors."""
    P = np.atleast_2d(np.asarray(member_probs, float))
    mean = P.mean(axis=0)
    total = _entropy(mean)
    aleatoric = _entropy(P).mean(axis=0)
    return {"mean": mean, "std": P.std(axis=0), "predictive_entropy": total,
            "aleatoric_entropy": aleatoric,
            "mutual_information": np.clip(total - aleatoric, 0, None)}


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# ----------------------------------------------------------------------------- estimators
class TreeEnsembleUncertainty:
    """Uses the individual trees of a fitted sklearn Pipeline(preprocess, RandomForest)."""

    name = "random_forest"

    def __init__(self, pipeline, max_trees: int | None = None):
        self.pipeline, self.max_trees = pipeline, max_trees

    def member_probs(self, frame) -> np.ndarray:
        pre, rf = self.pipeline[:-1], self.pipeline[-1]
        Xt = pre.transform(frame)
        trees = rf.estimators_[: self.max_trees] if self.max_trees else rf.estimators_
        if hasattr(Xt, "tocsr"):
            Xt = Xt.tocsr().astype(np.float32)
        else:
            Xt = np.asarray(Xt, dtype=np.float32)
        return np.stack([t.predict_proba(Xt, check_input=False)[:, 1] for t in trees])


class DeepEnsemble:
    """M bagged MLPClassifiers sharing one fitted preprocessor (Lakshminarayanan 2017)."""

    name = "deep_ensemble"

    def __init__(self, preprocessor, members: list):
        self.preprocessor, self.members = preprocessor, members

    def member_probs(self, frame) -> np.ndarray:
        X = _dense(self.preprocessor.transform(frame))
        return np.stack([m.predict_proba(X)[:, 1] for m in self.members])

    def predict_proba(self, frame) -> np.ndarray:
        p = self.member_probs(frame).mean(axis=0)
        return np.column_stack([1 - p, p])


class MCDropoutMLP:
    """Small numpy MLP trained with inverted dropout + Adam; MC dropout at inference."""

    name = "mc_dropout"

    def __init__(self, preprocessor=None, hidden=(64, 32), dropout: float = 0.2,
                 epochs: int = 15, batch_size: int = 512, lr: float = 1e-3,
                 l2: float = 1e-5, mc_samples: int = 30, seed: int = 0):
        self.preprocessor, self.hidden, self.p = preprocessor, tuple(hidden), dropout
        self.epochs, self.bs, self.lr, self.l2 = epochs, batch_size, lr, l2
        self.T, self.seed = mc_samples, seed
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

    # -- core ---------------------------------------------------------------------
    def _forward(self, X, rng=None, keep_cache=False):
        a, cache = X, []
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            h = np.maximum(z, 0)
            mask = None
            if rng is not None and self.p > 0:
                mask = (rng.random(h.shape) >= self.p) / (1 - self.p)
                h = h * mask
            if keep_cache:
                cache.append((a, z, mask))
            a = h
        logit = (a @ self.W[-1] + self.b[-1]).ravel()
        if keep_cache:
            cache.append((a, None, None))
        return 1 / (1 + np.exp(-np.clip(logit, -30, 30))), cache

    def fit(self, X, y, sample_weight=None, verbose: bool = False):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, float)
        sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)
        rng = np.random.default_rng(self.seed)
        sizes = [X.shape[1], *self.hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        params = self.W + self.b
        m = [np.zeros_like(q) for q in params]
        v = [np.zeros_like(q) for q in params]
        step, b1, b2 = 0, 0.9, 0.999
        for ep in range(self.epochs):
            order = rng.permutation(len(X))
            loss_sum = 0.0
            for s in range(0, len(X), self.bs):
                idx = order[s:s + self.bs]
                xb, yb, wb = X[idx], y[idx], sw[idx]
                prob, cache = self._forward(xb, rng, keep_cache=True)
                pc = np.clip(prob, 1e-7, 1 - 1e-7)
                loss_sum += float(-(wb * (yb * np.log(pc) + (1 - yb) * np.log(1 - pc))).sum())
                delta = (wb * (prob - yb) / wb.sum())[:, None]          # dL/dlogit
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                a_last = cache[-1][0]
                gW[-1] = a_last.T @ delta + self.l2 * self.W[-1]
                gb[-1] = delta.sum(0)
                d = delta @ self.W[-1].T
                for i in range(len(self.W) - 2, -1, -1):
                    a_in, z, mask = cache[i]
                    if mask is not None:
                        d = d * mask
                    d = d * (z > 0)
                    gW[i] = a_in.T @ d + self.l2 * self.W[i]
                    gb[i] = d.sum(0)
                    if i:
                        d = d @ self.W[i].T
                grads = gW + gb
                step += 1
                for j, (q, g) in enumerate(zip(params, grads)):
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    mh, vh = m[j] / (1 - b1 ** step), v[j] / (1 - b2 ** step)
                    q -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            if verbose:
                print(f"  mc-dropout epoch {ep + 1}/{self.epochs} loss={loss_sum / sw.sum():.4f}")
        return self

    # -- inference ------------------------------------------------------------------
    def _X(self, frame):
        X = frame if self.preprocessor is None else self.preprocessor.transform(frame)
        return np.asarray(_dense(X), np.float64)

    def member_probs(self, frame, T: int | None = None, seed: int = 123) -> np.ndarray:
        X = self._X(frame)
        rng = np.random.default_rng(seed)
        return np.stack([self._forward(X, rng)[0] for _ in range(T or self.T)])

    def predict_proba(self, frame) -> np.ndarray:
        p = self._forward(self._X(frame))[0]                # deterministic (dropout off)
        return np.column_stack([1 - p, p])


# ----------------------------------------------------------------------------- policy
DECISIONS = ("APPROVE", "MANUAL_REVIEW", "FLAG_FRAUD")


class ThreeWayPolicy:
    """Calibrated APPROVE / MANUAL_REVIEW / FLAG_FRAUD rule.

    Calibrated on held-out labeled data:
      * ``t_fraud``   : 5%-FPR threshold on the mean fraud score (business constraint)
      * ``t_low``     : lower edge of the grey zone, so that ``band_budget`` of traffic
                        falls in [t_low, t_fraud)
      * ``u_max``     : epistemic-uncertainty cut-off (quantile 1 - ``epistemic_budget``)
    Automatic flags are a subset of {score >= t_fraud}, so the automatic FPR <= 5%.
    """

    def __init__(self, target_fpr: float = 0.05, band_budget: float = 0.03,
                 epistemic_budget: float = 0.02, uncertainty_key: str = "mutual_information"):
        self.target, self.band, self.epi = target_fpr, band_budget, epistemic_budget
        self.key = uncertainty_key
        self.t_fraud = self.t_low = self.u_max = None

    def fit(self, metrics: dict, y) -> "ThreeWayPolicy":
        p, y = metrics["mean"], np.asarray(y)
        self.t_fraud = float(np.quantile(p[y == 0], 1 - self.target))
        below = p[p < self.t_fraud]
        share_below = len(below) / len(p)
        q = max(0.0, 1 - self.band / max(share_below, 1e-9))
        self.t_low = float(np.quantile(below, q)) if self.band > 0 else self.t_fraud
        u = metrics[self.key]
        self.u_max = float(np.quantile(u, 1 - self.epi)) if self.epi > 0 else float("inf")
        return self

    def decide(self, metrics: dict, t_fraud: float | None = None) -> np.ndarray:
        """Returns int codes: 0 = APPROVE, 1 = MANUAL_REVIEW, 2 = FLAG_FRAUD."""
        p, u = metrics["mean"], metrics[self.key]
        t_hi = self.t_fraud if t_fraud is None else t_fraud
        t_lo = self.t_low * (t_hi / self.t_fraud) if self.t_fraud else self.t_low
        out = np.ones(len(p), dtype=int)
        confident = u <= self.u_max
        out[(p < t_lo) & confident] = 0
        out[(p >= t_hi) & confident] = 2
        return out

    @classmethod
    def from_params(cls, d: dict) -> "ThreeWayPolicy":
        pol = cls(d["target_fpr"], d["band_budget"], d["epistemic_budget"], d["uncertainty_key"])
        pol.t_fraud, pol.t_low, pol.u_max = d["t_fraud"], d["t_low"], d["u_max"]
        return pol

    def params(self) -> dict:
        return {"t_fraud": self.t_fraud, "t_low": self.t_low, "u_max": self.u_max,
                "uncertainty_key": self.key, "band_budget": self.band,
                "epistemic_budget": self.epi, "target_fpr": self.target}


def evaluate_three_way(codes: np.ndarray, y: np.ndarray) -> dict:
    """Operational metrics; reviewers are assumed to resolve referred cases correctly."""
    y = np.asarray(y)
    auto = codes != 1
    flag, appr, rev = codes == 2, codes == 0, codes == 1
    fp_auto = int((flag & (y == 0)).sum())
    fn_auto = int((appr & (y == 1)).sum())
    n_auto = int(auto.sum())
    return {
        "review_rate": float(rev.mean()),
        "automation_rate": float(auto.mean()),
        "auto_error_rate": float((fp_auto + fn_auto) / max(n_auto, 1)),
        "auto_errors": fp_auto + fn_auto,
        "auto_false_positives": fp_auto, "auto_false_negatives": fn_auto,
        "auto_fpr": float(fp_auto / max((y == 0).sum(), 1)),
        "auto_recall": float((flag & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "recall_with_review": float(((flag | rev) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "review_fraud_rate": float(y[rev].mean()) if rev.any() else 0.0,
        "approve_fraud_rate": float(y[appr].mean()) if appr.any() else 0.0,
    }
