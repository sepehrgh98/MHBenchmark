import pandas as pd, argparse, os, json
from utils.common import write_jsonl, canon_text

def convert(df):
    for i, row in df.iterrows():
        qid = str(row["questionID"]).strip()
        title = str(row.get("questionTitle", "")).strip()
        qtext = str(row.get("questionText", "")).strip()
        qfull = (title + " " + qtext).strip()

        yield {
            "schema_version": "mhbench-v1",
            "id": f"CounselChat-{qid}-{i}",
            "dataset": "CounselChat",
            "subsource": str(row.get("topics", "")).strip(),
            "split": "TEST",
            "task": "generation",
            "subtype": "counsel_response",
            "group_id": qid,
            "input": {
                "question": canon_text(qfull)
            },
            "target": {
                "reply": canon_text(str(row.get("answerText", "")).strip()),
                "upvotes": int(row.get("upvotes", 0) or 0)
            },
            "meta": {
                "title": title,
                "therapist": str(row.get("therapistName", "")).strip(),
                "topics": [t.strip() for t in str(row.get("topics", "")).split(",") if t.strip()]
            }
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to counselchat_test.csv")
    ap.add_argument("--out", default="mhbench/benchmark/counselchat_test.jsonl")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = write_jsonl(args.out, convert(df))
    print(f"Wrote CounselChat TEST: {n} records -> {args.out}")
