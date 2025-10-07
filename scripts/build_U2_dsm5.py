import json
from pathlib import Path

RAW_DIR = Path("/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/DSM5/flat")
OUT_PATH = Path("/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U2.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def process_item(obj):
    variant = obj.get("task")
    base = {
        "task_id": "U2",
        "source_dataset": "DSM5",
        "instance_id": obj.get("uuid"),
        "input": {
            "prompt": obj.get("symptoms") or obj.get("vignette"),
            "choices": obj.get("options", [])
        },
        "reference": {
            "label": obj.get("correct_diagnosis") or obj.get("answer"),
            "explanation": obj.get("why_preferred") or obj.get("explanation")
        },
        "metadata": {
            "disorder": obj.get("disorder_context") or obj.get("disorder"),
            "section": obj.get("source_section") or obj.get("section"),
            "difficulty": obj.get("difficulty") if isinstance(obj.get("difficulty"), str)
                          else obj.get("difficulty", {}).get("level"),
            "task_variant": variant,
            "supporting_features": obj.get("supporting_features"),
            "misleading_cues": obj.get("misleading_cues"),
            "red_flags": obj.get("red_flags"),
            "sensitive": obj.get("sensitive"),
            "hallucination_flag": obj.get("hallucination_flag")
        }
    }
    return base

def main():
    files = [RAW_DIR / "3C.jsonl", RAW_DIR / "3D.jsonl"]
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
