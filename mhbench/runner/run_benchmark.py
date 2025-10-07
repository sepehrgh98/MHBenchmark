"""
MHBench Runner
--------------
Runs all benchmark tasks (U1–U7) across selected LLMs
and computes per-task metrics using MHBench metric modules.

Author: Sepehr Ghamari
"""

import json
import os
from pathlib import Path
from utils.llm_utils import query_llm
from tqdm import tqdm
import importlib

# List of benchmark tasks
TASKS = ["U1", "U2", "U3", "U4", "U5", "U6", "U7"]

# Directory paths
DATA_DIR = Path("benchmark_data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Available models
MODELS = [
    "gpt-4o",
    "claude-3-5-sonnet",
    "gemini-1.5-pro",
    "llama-3-70b",
    "mixtral-8x7b",
    "meditron-7b"
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_task(task_id: str, model_name: str):
    task_file = DATA_DIR / f"{task_id}.jsonl"
    data = load_jsonl(task_file)
    results = []

    for sample in tqdm(data, desc=f"{model_name} | {task_id}"):
        prompt = sample["input"]["prompt"]
        response = query_llm(prompt, backend=model_name)

        # Determine metric module
        if task_id in ["U1", "U2"]:
            metric_module = "metrics.knowledge_metrics"
        elif task_id == "U3":
            metric_module = "metrics.reasoning_coherence"
        elif task_id == "U4":
            metric_module = "metrics.empathy_metrics"
        elif task_id == "U5":
            metric_module = "metrics.toxicity_metrics"
        elif task_id == "U6":
            metric_module = "metrics.reasoning_coherence"
        elif task_id == "U7":
            metric_module = "metrics.toxicity_metrics"
        else:
            metric_module = None

        if metric_module:
            mod = importlib.import_module(metric_module)
            metrics = mod.compute(sample, response)
        else:
            metrics = {}

        results.append({
            "instance_id": sample["instance_id"],
            "model": model_name,
            "task_id": task_id,
            "response": response,
            "metrics": metrics
        })

    # Save results
    out_path = RESULTS_DIR / f"{model_name}_{task_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ Saved results to {out_path}")


if __name__ == "__main__":
    for model in MODELS:
        for task in TASKS:
            run_task(task, model)
