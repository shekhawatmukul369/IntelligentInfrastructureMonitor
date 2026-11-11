# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest infrastructure logs into Delta Lake
# MAGIC Set `source_path` to a CSV uploaded to a Unity Catalog Volume or DBFS.
# MAGIC This notebook creates Bronze (raw) and Silver (validated) Delta tables.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType

# Widgets make the notebook reusable across workspaces/environments.
dbutils.widgets.text("source_path", "/Volumes/main/observability/raw/infrastructure_logs.csv")
dbutils.widgets.text("target_schema", "main.observability")
dbutils.widgets.text("bronze_table_name", "bronze_infrastructure_logs")
dbutils.widgets.text("silver_table_name", "silver_infrastructure_logs")

source_path = dbutils.widgets.get("source_path")
target_schema = dbutils.widgets.get("target_schema")
bronze_table = f"{target_schema}.{dbutils.widgets.get('bronze_table_name')}"
silver_table = f"{target_schema}.{dbutils.widgets.get('silver_table_name')}"

# COMMAND ----------

raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(source_path)
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.input_file_name())
)

required = {"timestamp", "host", "service", "level", "message"}
missing = sorted(required.difference(set(raw.columns)))
if missing:
    raise ValueError(f"Input CSV is missing required columns: {missing}")

# Save raw rows first so that validation/reprocessing remains auditable.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")
(raw.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(bronze_table))

# COMMAND ----------

silver = raw
silver = silver.withColumn("timestamp", F.to_timestamp("timestamp"))
for col in ["cpu_usage", "memory_usage", "disk_usage", "network_in_mbps", "network_out_mbps", "response_time_ms"]:
    if col in silver.columns:
        silver = silver.withColumn(col, F.col(col).cast(DoubleType()))
if "error_count" in silver.columns:
    silver = silver.withColumn("error_count", F.col("error_count").cast(IntegerType()))
if "is_anomaly" in silver.columns:
    silver = silver.withColumn("is_anomaly", F.col("is_anomaly").cast(IntegerType()))

silver = (
    silver
    .filter(F.col("timestamp").isNotNull())
    .filter(F.col("host").isNotNull() & F.col("service").isNotNull())
    .withColumn("level", F.upper(F.coalesce(F.col("level"), F.lit("INFO"))))
    .withColumn("message", F.coalesce(F.col("message").cast("string"), F.lit("")))
    .dropDuplicates(["timestamp", "host", "service", "level", "message"])
)

(silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_table))

print(f"Bronze table: {bronze_table}")
print(f"Silver table: {silver_table}")
print(f"Validated Silver rows: {silver.count():,}")
display(silver.limit(20))
