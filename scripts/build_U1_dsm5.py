import json
from pathlib import Path

RAW_DIR = Path("/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/DSM5/flat")
OUT_PATH = Path("/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U1.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def process_item(obj):
    variant = obj.get("task")
    base = {
        "task_id": "U1",
        "source_dataset": "DSM5",
        "instance_id": obj.get("uuid"),
        "input": {
            "prompt": obj.get("question") or obj.get("symptoms"),
            "choices": obj.get("options", [])
        },
        "reference": {
            "label": obj.get("correct_answer") or obj.get("answer"),
            "explanation": obj.get("why_correct") or obj.get("explanation")
        },
        "metadata": {
            "disorder": obj.get("disorder") or obj.get("source_disorder"),
            "section": obj.get("section"),
            "difficulty": obj.get("difficulty"),
            "task_variant": variant,
            "sensitive": obj.get("sensitive"),
            "hallucination_flag": obj.get("hallucination_flag"),
            "evidence_quote": obj.get("evidence_quote")
        }
    }
    return base


def main():
    # Load 3A and 3B files (MC-QA and Differential Diagnosis)
    files = [RAW_DIR / "3A.jsonl", RAW_DIR / "3B.jsonl"]

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for file in files:
            if not file.exists():
                print(f"⚠️ Missing file: {file}")
                continue
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    unified = process_item(obj)
                    fout.write(json.dumps(unified, ensure_ascii=False) + "\n")

    print(f"✅ Saved {OUT_PATH}")

if __name__ == "__main__":
    main()