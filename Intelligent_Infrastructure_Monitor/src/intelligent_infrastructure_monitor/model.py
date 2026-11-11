"""Train, tune, persist, and use an Isolation Forest anomaly detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .features import FEATURE_COLUMNS, build_features
from .maintenance import train_maintenance_model

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def _fit_candidate(
    x_train_normal: np.ndarray,
    contamination: float,
    n_estimators: int,
    max_samples: str | int | float,
    random_state: int,
) -> tuple[RobustScaler, IsolationForest]:
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x_train_normal)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=1.0,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_scaled)
    return scaler, model


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_model(
    logs: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    if logs.empty or len(logs) < 30:
        raise ValueError("At least 30 log rows are required to train the model.")

    x = build_features(logs).to_numpy(dtype=float)
    has_labels = "is_anomaly" in logs.columns and logs["is_anomaly"].notna().any()
    if has_labels:
        y = pd.to_numeric(logs["is_anomaly"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
        if len(np.unique(y)) < 2:
            has_labels = False
    else:
        y = np.zeros(len(logs), dtype=int)

    candidates: list[dict[str, Any]] = []
    best_config: dict[str, Any] = {
        "contamination": 0.06,
        "n_estimators": 200,
        "max_samples": "auto",
    }
    validation_metrics: dict[str, float] | None = None
    test_metrics: dict[str, float] | None = None

    if has_labels and np.sum(y == 1) >= 5 and np.sum(y == 0) >= 30:
        indices = np.arange(len(y))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.20, random_state=random_state, stratify=y
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.25, random_state=random_state, stratify=y[train_indices]
        )
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # Model candidates are intentionally small enough for a capstone laptop.
        contamination_values = [0.03, 0.05, 0.07, 0.10]
        estimator_values = [150, 250]
        for contamination in contamination_values:
            for n_estimators in estimator_values:
                scaler, candidate_model = _fit_candidate(
                    x_train[y_train == 0], contamination, n_estimators, "auto", random_state
                )
                val_pred = (candidate_model.predict(scaler.transform(x_val)) == -1).astype(int)
                scores = _metrics(y_val, val_pred)
                config = {
                    "contamination": contamination,
                    "n_estimators": n_estimators,
                    "max_samples": "auto",
                    **scores,
                }
                candidates.append(config)

        best = max(candidates, key=lambda item: (item["f1"], item["recall"], item["precision"]))
        best_config = {key: best[key] for key in ("contamination", "n_estimators", "max_samples")}
        validation_metrics = {key: best[key] for key in ("accuracy", "precision", "recall", "f1")}

        # Independent hold-out evaluation with the selected configuration.
        scaler_eval, model_eval = _fit_candidate(
            x_train[y_train == 0],
            float(best_config["contamination"]),
            int(best_config["n_estimators"]),
            best_config["max_samples"],
            random_state,
        )
        test_pred = (model_eval.predict(scaler_eval.transform(x[test_indices])) == -1).astype(int)
        test_metrics = _metrics(y[test_indices], test_pred)

        final_fit_rows = x[y == 0]
        training_mode = "unsupervised_on_normal_rows_with_labelled_validation"
    else:
        final_fit_rows = x
        training_mode = "unsupervised_without_reliable_labels"

    final_scaler, final_model = _fit_candidate(
        final_fit_rows,
        float(best_config["contamination"]),
        int(best_config["n_estimators"]),
        best_config["max_samples"],
        random_state,
    )

    maintenance_bundle, maintenance_report = train_maintenance_model(logs, random_state=random_state)

    bundle = {
        "scaler": final_scaler,
        "model": final_model,
        "maintenance": maintenance_bundle,
        "maintenance_report": maintenance_report,
        "feature_columns": FEATURE_COLUMNS,
        "config": best_config,
        "training_mode": training_mode,
        "training_rows": int(len(logs)),
        "normal_training_rows": int(len(final_fit_rows)),
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    report = {
        "model_path": str(path),
        "training_mode": training_mode,
        "training_rows": int(len(logs)),
        "normal_training_rows": int(len(final_fit_rows)),
        "anomaly_rows_in_dataset": int(np.sum(y == 1)) if has_labels else None,
        "selected_config": best_config,
        "validation_metrics": validation_metrics,
        "holdout_metrics": test_metrics,
        "candidate_results": candidates,
        "maintenance_model": maintenance_report,
    }
    metrics_path = path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train one first with: iim train --input data/sample_infrastructure_logs.csv"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"scaler", "model"}.issubset(bundle):
        raise ValueError(f"The model file at {path} is not a valid IIM model bundle.")
    return bundle


def score_logs(logs: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of logs with anomaly scores, predictions, and severity."""
    if logs.empty:
        result = logs.copy()
        result["anomaly_score"] = pd.Series(dtype=float)
        result["is_anomaly_predicted"] = pd.Series(dtype=int)
        result["severity"] = pd.Series(dtype=str)
        return result

    features = build_features(logs)
    scaled = bundle["scaler"].transform(features.to_numpy(dtype=float))
    model = bundle["model"]
    predictions = (model.predict(scaled) == -1).astype(int)
    decision = model.decision_function(scaled)
    # A larger score means more unusual. Keep raw values for auditability.
    result = logs.copy()
    result["anomaly_score"] = (-decision).astype(float)
    result["is_anomaly_predicted"] = predictions

    def severity_for(row: pd.Series) -> str:
        if int(row["is_anomaly_predicted"]) == 0:
            return "Normal"

        def number_or_zero(value: Any) -> float:
            parsed = pd.to_numeric(value, errors="coerce")
            return 0.0 if pd.isna(parsed) else float(parsed)

        cpu = number_or_zero(row.get("cpu_usage", 0))
        memory = number_or_zero(row.get("memory_usage", 0))
        disk = number_or_zero(row.get("disk_usage", 0))
        level = str(row.get("level", "")).upper()
        errors = number_or_zero(row.get("error_count", 0))
        if cpu >= 95 or memory >= 95 or disk >= 98 or level in {"CRITICAL", "FATAL"} or errors >= 10:
            return "Critical"
        if cpu >= 85 or memory >= 85 or disk >= 90 or level == "ERROR" or errors >= 5:
            return "High"
        return "Medium"

    result["severity"] = result.apply(severity_for, axis=1)

    maintenance_bundle = bundle.get("maintenance")
    if maintenance_bundle is not None:
        maintenance_probability = maintenance_bundle["model"].predict_proba(features.to_numpy(dtype=float))[:, 1]
        threshold = float(maintenance_bundle.get("threshold", 0.5))
        result["failure_risk_probability"] = maintenance_probability.astype(float)
        result["maintenance_alert"] = (maintenance_probability >= threshold).astype(int)
        result["maintenance_risk"] = pd.cut(
            maintenance_probability,
            bins=[-np.inf, 0.30, 0.60, np.inf],
            labels=["Low", "Moderate", "High"],
            include_lowest=True,
        ).astype(str)
    else:
        result["failure_risk_probability"] = np.nan
        result["maintenance_alert"] = 0
        result["maintenance_risk"] = "Unavailable"
    return result
