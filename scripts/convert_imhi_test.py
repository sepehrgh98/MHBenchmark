import os, argparse, pandas as pd, re
from utils.common import write_jsonl, canon_text, hash_text_16

def parse_file(path, subsource):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx",".xls")) else pd.read_csv(path)

    for i, row in df.iterrows():
        query = str(row.get("query") or "").strip()
        resp  = str(row.get("gpt-3.5-turbo") or "").strip()

        if not query or not resp:
            continue

        text = canon_text(query)

        # Split response into label + rationale
        label = "Unknown"
        rationale = resp
        parts = re.split(r"(?i)reasoning\s*:\s*", resp, maxsplit=1)

        # first part → label (strip period at end if exists)
        first_part = parts[0].strip().rstrip(".")
        if first_part:
            label = first_part.capitalize()

        # rationale = second part if exists
        if len(parts) == 2:
            rationale = parts[1].strip()

        gid = hash_text_16(text)

        yield {
            "schema_version": "mhbench-v1",
            "id": f"IMHI-{subsource}-TEST-{i:05d}",
            "dataset": "IMHI",
            "subsource": subsource,
            "split": "TEST",
            "task": "classification",
            "subtype": "symptom2dx",
            "group_id": gid,
            "input": {"text": text},
            "choices": [],  # optional, can be filled with global label set later
            "target": {"label": label, "rationale": rationale},
            "meta": {"source_path": path}
        }

def iter_imhi_test(root_dir: str):
    for fn in os.listdir(root_dir):
        if fn.lower().endswith((".csv",".xlsx",".xls")):
            path = os.path.join(root_dir, fn)
            subsource = os.path.splitext(fn)[0]   # filename as subsource
            yield from parse_file(path, subsource)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imhi-test-root", required=True, help="Folder containing test CSV/Excel files")
    ap.add_argument("--out", default="mhbench/benchmark/imhi_test.jsonl")
    args = ap.parse_args()

    n = write_jsonl(args.out, iter_imhi_test(args.imhi_test_root))
    print(f"Wrote IMHI TEST: {n} records -> {args.out}")
