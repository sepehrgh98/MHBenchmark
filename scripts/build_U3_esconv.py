import json
from uuid import uuid4

INPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/ESConv/esconv_clean/esconv_test_clean.jsonl"
OUTPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U3.jsonl"

def build_summary(emotion, problem):
    return f"The seeker feels {emotion} due to {problem}."

u3_entries = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        instance_id = f"esconv_{i:05d}"

        entry = {
            "task_id": "U3",
            "source_dataset": "ESConv",
            "instance_id": instance_id,
            "input": {
                "prompt": d["situation"].strip(),
                "dialog_context": d["dialog"][:3]
            },
            "reference": {
                "emotion_label": d["emotion_type"].strip().lower(),
                "problem_label": d["problem_type"].strip().lower(),
                "summary": build_summary(
                    d["emotion_type"].lower(), 
                    d["problem_type"].lower()
                )
            },
            "metadata": {
                "difficulty": "moderate",
                "task_variant": "emotional_understanding",
                "sensitive": True,
                "hallucination_flag": "none"
            }
        }
        u3_entries.append(entry)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for e in u3_entries:
        f.write(json.dumps(e) + "\n")

print(f"✅ Saved {len(u3_entries)} entries to {OUTPUT_PATH}")
