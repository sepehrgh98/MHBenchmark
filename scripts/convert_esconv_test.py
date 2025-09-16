import json, argparse
from utils.common import write_jsonl, canon_text

def load_esconv_test(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            cid = ex.get("conversation_id", f"conv_{i}")
            topic = ex.get("problem_type") or ex.get("emotion_type") or ""
            dialog = ex.get("dialog", [])

            # --- Step 1: find first seeker utterance ---
            seeker_first_idx = next(
                (j for j, d in enumerate(dialog) if d["speaker"].lower() == "usr"),
                None
            )
            if seeker_first_idx is None:
                continue
            seeker_first = dialog[seeker_first_idx]

            # --- Step 2: find first supporter reply *after* seeker ---
            supporter_first = None
            for d in dialog[seeker_first_idx+1:]:
                if d["speaker"].lower() == "sys":
                    supporter_first = d
                    break
            if not supporter_first:
                continue

            # --- Step 3: build canonical record ---
            yield {
                "schema_version": "mhbench-v1",
                "id": f"ESConv-{cid}",
                "dataset": "ESConv",
                "subsource": "success",          # HuggingFace clean split = effective dialogues
                "split": "TEST",
                "task": "dialogue",
                "subtype": "empathy",
                "group_id": cid,
                "input": {
                    "messages": [
                        {"role": "seeker", "text": canon_text(seeker_first["text"])}
                    ]
                },
                "target": {
                    "reply": canon_text(supporter_first["text"]),
                    "label": "effective"
                },
                "meta": {
                    "topic": topic,
                    "strategies": [supporter_first.get("strategy")] if supporter_first.get("strategy") else []
                }
            }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esconv-test", required=True, help="Path to esconv_test_clean.jsonl")
    ap.add_argument("--out", default="mhbench/benchmark/esconv_test.jsonl")
    args = ap.parse_args()

    n = write_jsonl(args.out, load_esconv_test(args.esconv_test))
    print(f"Wrote ESConv TEST: {n} records -> {args.out}")
