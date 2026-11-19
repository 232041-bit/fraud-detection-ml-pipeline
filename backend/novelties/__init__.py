"""Research novelties added on top of the baseline BAF fraud-detection pipeline.

Novelty 1 - ``drift``:        Drift-Aware Fraud Detection Module (PSI / KS monitoring +
                               fairness-aware adaptive recalibration of the 5%-FPR threshold).
Novelty 2 - ``uncertainty``:  Uncertainty-Aware Fraud Detection (deep ensembles, Monte Carlo
                               dropout, tree-ensemble disagreement, predictive entropy) with a
                               three-way decision policy: APPROVE / MANUAL_REVIEW / FLAG_FRAUD.
"""
from .drift import (DriftMonitor, AdaptiveThresholdPolicy, population_stability_index,
                    age_group, predictive_equality)
from .uncertainty import (MCDropoutMLP, DeepEnsemble, TreeEnsembleUncertainty,
                          ThreeWayPolicy, uncertainty_metrics)

__all__ = ["DriftMonitor", "AdaptiveThresholdPolicy", "population_stability_index",
           "age_group", "predictive_equality", "MCDropoutMLP", "DeepEnsemble",
           "TreeEnsembleUncertainty", "ThreeWayPolicy", "uncertainty_metrics"]
