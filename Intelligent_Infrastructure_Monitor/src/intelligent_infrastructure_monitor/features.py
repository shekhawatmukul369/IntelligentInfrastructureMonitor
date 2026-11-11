"""Feature engineering for structured infrastructure logs."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_in_mbps",
    "network_out_mbps",
    "error_count",
    "response_time_ms",
    "log_level_code",
    "message_length",
    "keyword_exception",
    "keyword_timeout",
    "keyword_oom",
    "keyword_disk_full",
    "keyword_connection",
    "keyword_restart",
    "keyword_failed",
    "hour_of_day",
    "day_of_week",
]

LEVEL_CODES = {
    "TRACE": 0,
    "DEBUG": 1,
    "INFO": 2,
    "WARN": 3,
    "WARNING": 3,
    "ERROR": 4,
    "CRITICAL": 5,
    "FATAL": 5,
}

KEYWORD_PATTERNS = {
    "keyword_exception": r"exception|traceback|stack trace",
    "keyword_timeout": r"timeout|timed out|deadline exceeded",
    "keyword_oom": r"out of memory|oom|memory exhausted|killed process",
    "keyword_disk_full": r"disk full|no space left|filesystem full",
    "keyword_connection": r"connection refused|connection reset|unreachable|broken pipe",
    "keyword_restart": r"restart|restarting|crash|terminated",
    "keyword_failed": r"failed|failure|error|unavailable",
}

NUMERIC_DEFAULTS = {
    "cpu_usage": 0.0,
    "memory_usage": 0.0,
    "disk_usage": 0.0,
    "network_in_mbps": 0.0,
    "network_out_mbps": 0.0,
    "error_count": 0.0,
    "response_time_ms": 0.0,
}


def _numeric_column(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype("float64")


def build_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Convert raw log rows into numeric model features.

    The function is deliberately tolerant of missing optional fields so that
    operators can score logs from different monitoring systems after mapping
    their column names to the documented input schema.
    """
    if not isinstance(logs, pd.DataFrame):
        raise TypeError("logs must be a pandas DataFrame")

    features = pd.DataFrame(index=logs.index)
    for column, default in NUMERIC_DEFAULTS.items():
        features[column] = _numeric_column(logs, column, default)

    if "level" in logs.columns:
        levels = logs["level"].fillna("INFO").astype(str).str.upper().str.strip()
    else:
        levels = pd.Series("INFO", index=logs.index)
    features["log_level_code"] = levels.map(LEVEL_CODES).fillna(2).astype("float64")

    if "message" in logs.columns:
        messages = logs["message"].fillna("").astype(str).str.lower()
    else:
        messages = pd.Series("", index=logs.index, dtype="object")
    features["message_length"] = messages.str.len().astype("float64")
    for feature_name, pattern in KEYWORD_PATTERNS.items():
        features[feature_name] = messages.str.contains(pattern, regex=True, na=False).astype("float64")

    if "timestamp" in logs.columns:
        timestamps = pd.to_datetime(logs["timestamp"], errors="coerce", utc=True)
        features["hour_of_day"] = timestamps.dt.hour.fillna(0).astype("float64")
        features["day_of_week"] = timestamps.dt.dayofweek.fillna(0).astype("float64")
    else:
        features["hour_of_day"] = 0.0
        features["day_of_week"] = 0.0

    # Keep columns in a stable order and prevent infinities from breaking fit/score.
    features = features.reindex(columns=FEATURE_COLUMNS)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def validate_log_columns(columns: Iterable[str]) -> list[str]:
    """Return missing required columns; labels and metrics are optional except timestamp."""
    required = {"timestamp", "host", "service", "level", "message"}
    return sorted(required.difference(set(columns)))
