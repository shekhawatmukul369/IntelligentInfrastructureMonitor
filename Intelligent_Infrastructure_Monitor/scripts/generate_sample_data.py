

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOSTS = ["web-01", "web-02", "api-01", "worker-01", "db-01", "cache-01"]
SERVICES = {
    "web-01": "nginx", "web-02": "nginx", "api-01": "payments-api",
    "worker-01": "queue-worker", "db-01": "postgres", "cache-01": "redis",
}


def generate_logs(rows: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-10-30T00:00:00Z")
    timestamps = start + pd.to_timedelta(np.sort(rng.uniform(0, 24 * 60, rows)), unit="m")
    host = rng.choice(HOSTS, size=rows)
    service = [SERVICES[item] for item in host]

    cpu = np.clip(rng.normal(38, 13, rows), 3, 82)
    memory = np.clip(rng.normal(48, 12, rows), 10, 82)
    disk = np.clip(rng.normal(54, 15, rows), 8, 88)
    net_in = np.clip(rng.normal(42, 18, rows), 0, 120)
    net_out = np.clip(rng.normal(35, 16, rows), 0, 110)
    errors = rng.poisson(0.45, rows).astype(float)
    latency = np.clip(rng.normal(180, 65, rows), 20, 650)
    level = np.full(rows, "INFO", dtype=object)
    messages = np.full(rows, "Request processed successfully", dtype=object)
    labels = np.zeros(rows, dtype=int)

    anomaly_count = max(50, int(rows * 0.075))
    anomaly_indices = rng.choice(rows, size=anomaly_count, replace=False)
    patterns = rng.choice(["cpu", "memory", "disk", "network", "error"], size=anomaly_count)
    for idx, pattern in zip(anomaly_indices, patterns):
        labels[idx] = 1
        if pattern == "cpu":
            cpu[idx] = rng.uniform(91, 100)
            latency[idx] = rng.uniform(900, 2600)
            level[idx] = "WARN" if rng.random() < 0.5 else "ERROR"
            messages[idx] = "High CPU utilization; request processing delayed"
        elif pattern == "memory":
            memory[idx] = rng.uniform(91, 100)
            errors[idx] = rng.integers(5, 18)
            level[idx] = "ERROR"
            messages[idx] = "Out of memory warning; worker process restarted"
        elif pattern == "disk":
            disk[idx] = rng.uniform(94, 100)
            errors[idx] = rng.integers(3, 12)
            level[idx] = "CRITICAL"
            messages[idx] = "Filesystem nearly full; no space left for write operation"
        elif pattern == "network":
            latency[idx] = rng.uniform(1200, 5000)
            net_in[idx] = rng.uniform(160, 500)
            net_out[idx] = rng.uniform(140, 450)
            level[idx] = "ERROR"
            messages[idx] = "Upstream connection timeout; connection reset by peer"
        else:
            errors[idx] = rng.integers(8, 30)
            latency[idx] = rng.uniform(800, 4000)
            level[idx] = "ERROR" if rng.random() < 0.75 else "CRITICAL"
            messages[idx] = "Service request failed; dependency unavailable"

    # Synthetic future-maintenance outcome for demonstrating the supervised
    # predictive-maintenance branch. In a real project this label must come
    # from incident/work-order history and a defined 24-hour prediction window.
    failure_count = max(35, int(rows * 0.045))
    failure_indices = rng.choice(rows, size=failure_count, replace=False)
    failure_labels = np.zeros(rows, dtype=int)
    failure_labels[failure_indices] = 1
    for idx in failure_indices:
        cpu[idx] = max(cpu[idx], rng.uniform(79, 91))
        memory[idx] = max(memory[idx], rng.uniform(78, 91))
        disk[idx] = max(disk[idx], rng.uniform(82, 94))
        errors[idx] = max(errors[idx], int(rng.integers(3, 9)))
        latency[idx] = max(latency[idx], rng.uniform(550, 1600))
        if level[idx] == "INFO":
            level[idx] = "WARN"
        if messages[idx] == "Request processed successfully":
            messages[idx] = "Repeated resource pressure detected; maintenance threshold approaching"

    df = pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "host": host,
        "service": service,
        "level": level,
        "message": messages,
        "cpu_usage": np.round(cpu, 2),
        "memory_usage": np.round(memory, 2),
        "disk_usage": np.round(disk, 2),
        "network_in_mbps": np.round(net_in, 2),
        "network_out_mbps": np.round(net_out, 2),
        "error_count": errors.astype(int),
        "response_time_ms": np.round(latency, 2),
        "is_anomaly": labels,
        "failure_within_24h": failure_labels,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1600)
    parser.add_argument("--output", default="data/sample_infrastructure_logs.csv")
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("--rows must be at least 100")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_logs(args.rows).to_csv(output, index=False)
    print(f"Generated {args.rows} rows at {output.resolve()}")


if __name__ == "__main__":
    main()
