import csv
import json
import re
from uuid import uuid4

def clean_text(text):
    """Remove HTML tags, entities, and extra spaces."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities like &nbsp; or &#34;
    text = re.sub(r'&nbsp;|&#160;', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    # Replace multiple spaces/newlines with one space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()

# ---------- File paths ----------
INPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/CounselChat/clean/counselchat_test.csv"
OUTPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U6.jsonl"

entries = []

# ---------- Conversion ----------
with open(INPUT_PATH, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Clean and normalize
        question = clean_text(row.get("questionText", ""))
        answer = clean_text(row.get("answerText", ""))
        topic = clean_text(row.get("topics", "General Counseling"))
        
        # Skip incomplete rows
        if not question or not answer:
            continue

        entry = {
            "task_id": "U6",
            "source_dataset": "CounselChat",
            "instance_id": str(uuid4()),
            "input": {
                "prompt": question
            },
            "reference": {
                "response_text": answer
            },
            "metadata": {
                "topic": topic,
                "difficulty": "moderate",
                "task_variant": "therapeutic_reasoning",
                "sensitive": True,
                "hallucination_flag": "none"
            }
        }
        entries.append(entry)

# ---------- Save ----------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"✅ Created {len(entries)} cleaned U6 entries and saved to {OUTPUT_PATH}")
