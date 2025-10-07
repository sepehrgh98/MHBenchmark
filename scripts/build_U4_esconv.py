import json, uuid

INPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/ESConv/esconv_clean/esconv_test_clean.jsonl"
OUTPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U4.jsonl"

entries = []
for line in open(INPUT_PATH, "r", encoding="utf-8"):
    data = json.loads(line)
    dialog = data["dialog"]
    for i in range(len(dialog) - 1):
        if dialog[i]["speaker"] == "usr" and dialog[i + 1]["speaker"] == "sys":
            instance_id = f"{uuid.uuid4()}"
            entry = {
                "task_id": "U4",
                "source_dataset": "ESConv",
                "instance_id": instance_id,
                "input": {
                    "prompt": f"User: {dialog[i]['text']}",
                    "dialog_context": dialog[max(0, i - 3):i]
                },
                "reference": {
                    "response_text": dialog[i + 1]["text"]
                },
                "metadata": {
                    "strategy_label": dialog[i + 1].get("strategy", "Unknown"),
                    "difficulty": "moderate",
                    "task_variant": "supportive_generation",
                    "sensitive": True,
                    "hallucination_flag": "none"
                }
            }
            entries.append(entry)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"✅ Created {len(entries)} U4 entries")
