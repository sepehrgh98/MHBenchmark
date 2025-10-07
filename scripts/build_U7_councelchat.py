import csv
import json
import re
import random
from uuid import uuid4
from pathlib import Path

# ---------- Utility ----------
def clean_text(text):
    """Remove HTML tags, entities, and extra spaces."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- File paths ----------
COUNSELCHAT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/CounselChat/clean/counselchat_test.csv"
ESCONV_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/raw/ESConv/esconv_clean/esconv_test_clean.jsonl"
OUTPUT_PATH = "/speed-scratch/se_gham/MHBenchmark/MHBenchmark/mhbench/benchmark/U7.jsonl"

entries = []

# ===============================================================
# 🩵 1. Load and process CounselChat (CSV)
# ===============================================================
with open(COUNSELCHAT_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = clean_text(row.get("questionText", ""))
        answer = clean_text(row.get("answerText", ""))
        topic = clean_text(row.get("topics", "General Counseling"))

        if not question or not answer:
            continue

        entry = {
            "task_id": "U7",
            "source_dataset": "CounselChat",
            "instance_id": str(uuid4()),
            "input": {"prompt": question},
            "reference": {
                "response_text": answer,
                "confidence": None  # placeholder for calibration scoring
            },
            "metadata": {
                "topic": topic,
                "source_type": "qa_pair",
                "difficulty": "moderate",
                "task_variant": "safety_helpfulness",
                "sensitive": True,
                "hallucination_flag": "none",
            },
        }
        entries.append(entry)

print(f"✅ Loaded {len(entries)} entries from CounselChat")

# ===============================================================
# 💜 2. Load and process ESConv (JSONL)
# ===============================================================
esconv_count = 0
with open(ESCONV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            convo = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "dialog" not in convo or not isinstance(convo["dialog"], list):
            continue

        dialog = convo["dialog"]
        context_buffer = []

        for turn in dialog:
            text = clean_text(turn.get("text", ""))
            speaker = turn.get("speaker", "").lower()
            if not text:
                continue

            if speaker == "usr":
                context_buffer.append(text)
            elif speaker == "sys":
                context = " ".join(context_buffer[-3:])  # last 3 user turns
                response = text
                strategy = clean_text(turn.get("strategy", "general_support"))
                if not context or not response:
                    continue

                entry = {
                    "task_id": "U7",
                    "source_dataset": "ESConv",
                    "instance_id": str(uuid4()),
                    "input": {"prompt": context},
                    "reference": {
                        "response_text": response,
                        "confidence": None  # placeholder for calibration scoring
                    },
                    "metadata": {
                        "strategy": strategy,
                        "source_type": "dialogue_turn",
                        "difficulty": "moderate",
                        "task_variant": "safety_diversity",
                        "sensitive": True,
                        "hallucination_flag": "none",
                    },
                }
                entries.append(entry)
                esconv_count += 1

print(f"✅ Loaded {esconv_count} entries from ESConv")

# ===============================================================
# 🔀 3. Shuffle entries for randomness
# ===============================================================
random.shuffle(entries)
print("🔀 Shuffled merged dataset to mix sources randomly")

# ===============================================================
# 💾 4. Save merged dataset
# ===============================================================
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"✅ Created {len(entries)} total U7 entries (shuffled) and saved to {OUTPUT_PATH}")
