"""High-level CSV ingestion, model training, scoring, and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .explain import generate_explanation
from .model import DEFAULT_MODEL_PATH, load_model, score_logs, train_model


def read_logs(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {path}")
    logs = pd.read_csv(path)
    if logs.empty:
        raise ValueError(f"Input CSV contains no rows: {path}")
    return logs


def analyze_logs(
    logs: pd.DataFrame,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_dir: str | Path = "reports",
    explain: bool = False,
    max_explanations: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = load_model(model_path)
    scored = score_logs(logs, bundle)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    anomaly_rows = scored[scored["is_anomaly_predicted"] == 1].copy()
    explanation_records: list[dict[str, Any]] = []
    if explain:
        selected = anomaly_rows.sort_values("anomaly_score", ascending=False).head(max_explanations)
        for index, row in selected.iterrows():
            explanation = generate_explanation(row)
            explanation_records.append({
                "row_index": int(index) if isinstance(index, (int,)) else str(index),
                "host": str(row.get("host", "")),
                "service": str(row.get("service", "")),
                "severity": str(row.get("severity", "")),
                **explanation,
            })

    csv_path = output / "anomaly_results.csv"
    scored.to_csv(csv_path, index=False)
    report = {
        "input_rows": int(len(scored)),
        "anomalies_detected": int(len(anomaly_rows)),
        "normal_rows": int(len(scored) - len(anomaly_rows)),
        "severity_counts": anomaly_rows["severity"].value_counts().to_dict(),
        "maintenance_alerts": int(scored["maintenance_alert"].sum()) if "maintenance_alert" in scored.columns else 0,
        "maintenance_risk_counts": scored["maintenance_risk"].value_counts().to_dict() if "maintenance_risk" in scored.columns else {},
        "maintenance_model": bundle.get("maintenance_report", {}),
        "model_config": bundle.get("config", {}),
        "model_training_mode": bundle.get("training_mode", "unknown"),
        "output_csv": str(csv_path),
        "explanations": explanation_records,
    }
    (output / "analysis_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if explanation_records:
        (output / "explanations.json").write_text(json.dumps(explanation_records, indent=2, default=str), encoding="utf-8")
    return scored, report


def train_from_csv(
    csv_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    return train_model(read_logs(csv_path), model_path=model_path)
