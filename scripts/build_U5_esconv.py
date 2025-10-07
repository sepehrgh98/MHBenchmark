import json, uuid

INPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/ESConv/esconv_clean/esconv_test_clean.jsonl"
OUTPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U5.jsonl"

entries = []
for line in open(INPUT_PATH, "r", encoding="utf-8"):
    data = json.loads(line)
    dialog = data["dialog"]
    for i, turn in enumerate(dialog):
        if turn["speaker"] == "sys":
            user_context = [d for d in dialog[max(0, i-3):i] if d["speaker"] == "usr"]
            entry = {
                "task_id": "U5",
                "source_dataset": "ESConv",
                "instance_id": str(uuid.uuid4()),
                "input": {
                    "prompt": turn["text"],
                    "dialog_context": user_context
                },
                "reference": {
                    "strategy_label": turn.get("strategy", "Unknown")
                },
                "metadata": {
                    "difficulty": "easy",
                    "task_variant": "strategy_classification",
                    "sensitive": True,
                    "hallucination_flag": "none"
                }
            }
            entries.append(entry)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"✅ Created {len(entries)} U5 entries")
