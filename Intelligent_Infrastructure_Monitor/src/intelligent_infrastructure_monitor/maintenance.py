"""Optional supervised failure-risk model for predictive-maintenance labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS, build_features


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def train_maintenance_model(logs: pd.DataFrame, random_state: int = 42) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if "failure_within_24h" not in logs.columns:
        return None, {"status": "not_trained", "reason": "failure_within_24h label column is absent"}

    valid = pd.to_numeric(logs["failure_within_24h"], errors="coerce").notna()
    usable = logs.loc[valid].copy()
    y = pd.to_numeric(usable["failure_within_24h"], errors="coerce").astype(int).clip(0, 1).to_numpy()
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if len(y) < 100 or positives < 10 or negatives < 30:
        return None, {
            "status": "not_trained",
            "reason": "need at least 100 labelled rows, 10 positive labels, and 30 negative labels",
            "labelled_rows": int(len(y)),
            "positive_rows": positives,
            "negative_rows": negatives,
        }

    x = build_features(usable).to_numpy(dtype=float)
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.20, random_state=random_state, stratify=y
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.25, random_state=random_state, stratify=y[train_indices]
    )

    candidates: list[dict[str, Any]] = []
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    best: dict[str, Any] | None = None
    best_validation_f1 = -1.0
    for n_estimators in (150, 250):
        for max_depth in (None, 10):
            candidate = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )
            candidate.fit(x[train_indices], y[train_indices])
            val_prob = candidate.predict_proba(x[val_indices])[:, 1]
            for threshold in thresholds:
                metric = _classification_metrics(y[val_indices], val_prob, threshold)
                record = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "threshold": threshold,
                    **metric,
                }
                candidates.append(record)
                sort_key = (metric["f1"], metric["recall"], metric["precision"])
                if sort_key > (best_validation_f1, (best or {}).get("recall", -1), (best or {}).get("precision", -1)):
                    best_validation_f1 = metric["f1"]
                    best = record

    assert best is not None
    # Independent hold-out metric for the selected hyperparameters and threshold.
    holdout_model = RandomForestClassifier(
        n_estimators=int(best["n_estimators"]),
        max_depth=best["max_depth"],
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    holdout_model.fit(x[train_indices], y[train_indices])
    holdout_prob = holdout_model.predict_proba(x[test_indices])[:, 1]
    holdout_metrics = _classification_metrics(y[test_indices], holdout_prob, float(best["threshold"]))

    final_model = RandomForestClassifier(
        n_estimators=int(best["n_estimators"]),
        max_depth=best["max_depth"],
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(x, y)
    bundle = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(best["threshold"]),
        "target": "failure_within_24h",
    }
    report = {
        "status": "trained",
        "target": "failure_within_24h",
        "labelled_rows": int(len(y)),
        "positive_rows": positives,
        "negative_rows": negatives,
        "selected_config": {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "threshold": best["threshold"],
        },
        "validation_metrics": {key: best[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc") if key in best},
        "holdout_metrics": holdout_metrics,
        "candidate_results": candidates,
    }
    return bundle, report
