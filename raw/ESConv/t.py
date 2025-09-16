# clean_esconv.py
import os
import json

def clean_file(input_path, output_path):
    """
    Read a HuggingFace-exported ESConv JSONL/CSV (with nested "text" field)
    and write a clean JSONL (one conversation dict per line).
    """
    cleaned = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # HuggingFace JSONL export has {"text": "{...json string...}"}
                outer = json.loads(line)
                inner = json.loads(outer["text"])
                cleaned.append(inner)
            except Exception as e:
                print(f"⚠️ Skipped a line in {input_path}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Cleaned {len(cleaned)} conversations -> {output_path}")


def main():
    input_dir = "esconv_splits"     # where your raw HuggingFace exports are
    output_dir = "esconv_clean"     # new folder with cleaned files
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        input_path = os.path.join(input_dir, f"esconv_{split}.jsonl")
        output_path = os.path.join(output_dir, f"esconv_{split}_clean.jsonl")
        if os.path.exists(input_path):
            clean_file(input_path, output_path)
        else:
            print(f"⚠️ File not found: {input_path}")

if __name__ == "__main__":
    main()
