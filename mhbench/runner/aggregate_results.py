"""
MHBench Aggregator
------------------
Aggregates per-task results into one summary CSV for analysis.

Author: Sepehr Ghamari
"""

import json
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

def aggregate_results():
    records = []
    for file in RESULTS_DIR.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                base = {
                    "model": item["model"],
                    "task_id": item["task_id"],
                    "instance_id": item["instance_id"],
                }
                metrics = item.get("metrics", {})
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            base[f"{k}.{subk}"] = subv
                    else:
                        base[k] = v
                records.append(base)

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["model", "task_id"])
        .mean(numeric_only=True)
        .reset_index()
    )
    df.to_csv("results/mhbench_raw.csv", index=False)
    summary.to_csv("results/mhbench_summary.csv", index=False)
    print("✅ Aggregated results saved to results/mhbench_summary.csv")


if __name__ == "__main__":
    aggregate_results()
