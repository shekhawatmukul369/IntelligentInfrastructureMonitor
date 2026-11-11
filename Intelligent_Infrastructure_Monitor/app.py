

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intelligent_infrastructure_monitor.explain import generate_explanation  # noqa: E402
from intelligent_infrastructure_monitor.model import DEFAULT_MODEL_PATH, load_model, score_logs, train_model  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "sample_infrastructure_logs.csv"
MODEL_PATH = PROJECT_ROOT / DEFAULT_MODEL_PATH

st.set_page_config(page_title="Intelligent Infrastructure Monitor", page_icon="🖥️", layout="wide")
st.title("🖥️ Intelligent Infrastructure Monitor")
st.caption("Predictive maintenance signals · Anomaly detection · AI-assisted incident explanations")


def ensure_model() -> dict:
    if not MODEL_PATH.exists():
        if not DATA_PATH.exists():
            raise FileNotFoundError("Sample data is missing. Run scripts/generate_sample_data.py first.")
        training_logs = pd.read_csv(DATA_PATH)
        train_model(training_logs, model_path=MODEL_PATH)
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Upload infrastructure logs (CSV)", type=["csv"])
    st.markdown("**Required columns**")
    st.code("timestamp, host, service, level, message", language="text")
    st.markdown("**Optional metric columns**")
    st.code("cpu_usage, memory_usage, disk_usage, network_in_mbps, network_out_mbps, error_count, response_time_ms", language="text")
    st.caption("Metrics are numeric. Utilization columns use percentages from 0–100.")

try:
    model_bundle = ensure_model()
    raw_logs = pd.read_csv(upload) if upload is not None else load_sample()
    if raw_logs.empty:
        st.error("The selected CSV has no rows.")
        st.stop()
    scored = score_logs(raw_logs, model_bundle)
except Exception as exc:
    st.error(f"Could not load or score the log data: {exc}")
    st.stop()

anomalies = scored[scored["is_anomaly_predicted"] == 1].copy()
critical_count = int((anomalies["severity"] == "Critical").sum())
high_count = int((anomalies["severity"] == "High").sum())
maintenance_alerts = int(scored["maintenance_alert"].sum()) if "maintenance_alert" in scored.columns else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Log records", f"{len(scored):,}")
col2.metric("Detected anomalies", f"{len(anomalies):,}")
col3.metric("High / Critical", f"{high_count + critical_count:,}")
col4.metric("Failure-risk alerts", f"{maintenance_alerts:,}")
col5.metric("Normal records", f"{len(scored) - len(anomalies):,}")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Anomaly timeline")
    if "timestamp" in scored.columns:
        timeline = scored.copy()
        timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
        timeline = timeline.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not timeline.empty:
            timeline["status"] = timeline["is_anomaly_predicted"].map({0: "Normal", 1: "Anomaly"})
            fig = px.scatter(
                timeline, x="timestamp", y="anomaly_score", color="status",
                hover_data=[c for c in ["host", "service", "level", "severity"] if c in timeline.columns],
                title="Model anomaly score over time",
            )
            fig.update_layout(height=330, legend_title_text="Detection")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No parseable timestamps were found.")
with right:
    st.subheader("Predicted maintenance risk")
    if "maintenance_risk" in scored.columns and (scored["maintenance_risk"] != "Unavailable").any():
        risk_counts = scored["maintenance_risk"].value_counts().rename_axis("risk").reset_index(name="records")
        risk_fig = px.bar(risk_counts, x="risk", y="records", color="risk", title="Risk estimates for failure within 24 hours")
        risk_fig.update_layout(height=260, showlegend=False)
        st.plotly_chart(risk_fig, use_container_width=True)
    else:
        st.info("No failure_within_24h labels were available to train a predictive-maintenance model.")

    st.subheader("Detected severity")
    if anomalies.empty:
        st.success("No anomalies were flagged in this dataset.")
    else:
        severity_counts = anomalies["severity"].value_counts().rename_axis("severity").reset_index(name="records")
        fig = px.bar(severity_counts, x="severity", y="records", color="severity", title="Anomalies by severity")
        fig.update_layout(height=330, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Detected events")
if anomalies.empty:
    st.info("There are no detected anomalies to explain. Try the included sample data or upload another log export.")
else:
    display_columns = [c for c in ["timestamp", "host", "service", "level", "severity", "anomaly_score", "failure_risk_probability", "maintenance_risk", "cpu_usage", "memory_usage", "disk_usage", "message"] if c in anomalies.columns]
    anomalies = anomalies.sort_values("anomaly_score", ascending=False)
    st.dataframe(anomalies[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Root-cause assistant")
    options = anomalies.index.tolist()
    selected_index = st.selectbox(
        "Choose an anomaly to investigate",
        options=options,
        format_func=lambda idx: (
            f"{anomalies.loc[idx].get('host', 'host')} · "
            f"{anomalies.loc[idx].get('service', 'service')} · "
            f"{anomalies.loc[idx].get('severity', 'Unknown')} · row {idx}"
        ),
    )
    selected = anomalies.loc[selected_index]
    with st.expander("Evidence sent to the explanation component", expanded=False):
        st.json({k: (v.item() if hasattr(v, "item") else str(v) if isinstance(v, pd.Timestamp) else v)
                 for k, v in selected.to_dict().items()
                 if k in {"timestamp", "host", "service", "level", "message", "severity", "anomaly_score", "failure_risk_probability", "maintenance_risk", "cpu_usage", "memory_usage", "disk_usage", "error_count", "response_time_ms"}})
    if st.button("Explain selected anomaly", type="primary"):
        with st.spinner("Preparing evidence-based explanation..."):
            explanation = generate_explanation(selected)
        if explanation["source"].startswith("llm:"):
            st.caption(f"Generated using {explanation['source']}")
        else:
            st.caption("Local rule-based explanation · Configure LLM_API_KEY in .env to enable GenAI explanations")
        st.markdown(explanation["explanation"])

with st.expander("Model information"):
    st.json({
        "training_mode": model_bundle.get("training_mode"),
        "training_rows": model_bundle.get("training_rows"),
        "normal_training_rows": model_bundle.get("normal_training_rows"),
        "selected_configuration": model_bundle.get("config"),
        "maintenance_model": model_bundle.get("maintenance_report"),
        "trained_at_utc": model_bundle.get("trained_at_utc"),
    })
    st.caption("The detector identifies behaviour that differs from its training baseline. This is an anomaly signal, not proof that a failure will occur.")
