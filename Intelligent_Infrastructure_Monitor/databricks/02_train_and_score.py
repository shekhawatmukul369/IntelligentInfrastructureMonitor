# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train the custom detector and score Delta logs
# MAGIC Run notebook 01 first. Attach the project repository (or install its wheel)
# MAGIC so that `intelligent_infrastructure_monitor` is importable on the cluster.

# COMMAND ----------

import os
import sys
from pathlib import Path
from pyspark.sql import functions as F

# Update this widget to the workspace path where the project repository is checked out.
dbutils.widgets.text("project_src_path", "/Workspace/Repos/<your-user>/IntelligentInfrastructureMonitor/src")
dbutils.widgets.text("input_table", "main.observability.silver_infrastructure_logs")
dbutils.widgets.text("output_table", "main.observability.scored_infrastructure_logs")
dbutils.widgets.text("model_path", "/dbfs/FileStore/intelligent-infrastructure-monitor/model.joblib")

project_src_path = dbutils.widgets.get("project_src_path")
input_table = dbutils.widgets.get("input_table")
output_table = dbutils.widgets.get("output_table")
model_path = dbutils.widgets.get("model_path")

if project_src_path not in sys.path:
    sys.path.insert(0, project_src_path)

from intelligent_infrastructure_monitor.model import train_model, score_logs

# COMMAND ----------

source_sdf = spark.table(input_table)
# This reference implementation uses pandas/scikit-learn for the custom model.
# For very large log tables, train on a representative sample and score in batches.
logs_pdf = source_sdf.toPandas()
if len(logs_pdf) > 500_000:
    raise ValueError("More than 500,000 rows selected. Sample the Delta table before toPandas().")

Path(model_path).parent.mkdir(parents=True, exist_ok=True)
training_report = train_model(logs_pdf, model_path=model_path)
print("Training report:")
print(training_report)

# COMMAND ----------

import joblib
bundle = joblib.load(model_path)
scored_pdf = score_logs(logs_pdf, bundle)
scored_sdf = spark.createDataFrame(scored_pdf)
(scored_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table))

print(f"Scored {len(scored_pdf):,} rows into {output_table}")
print(f"Detected anomalies: {int(scored_pdf['is_anomaly_predicted'].sum()):,}")
display(scored_sdf.filter("is_anomaly_predicted = 1").orderBy(F.desc("anomaly_score")).limit(50))
