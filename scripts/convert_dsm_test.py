import argparse
from utils.common import read_jsonl, write_jsonl

def map_3a(path):
    for ex in read_jsonl(path):
        yield {
            "schema_version":"mhbench-v1",
            "id": f"DSM-3A-{ex['id']}",
            "dataset":"DSM","subsource":"3A",
            "split":"TEST",
            "task":"mcq","subtype":"knowledge",
            "group_id": ex["disorder"],
            "input": {"stem": ex["stem"], "options": ex["options"], "context": ex.get("context","")},
            "choices": ["A","B","C","D"][:len(ex["options"])],
            "target": {"label_idx": ex["answer_idx"], "label": ex["options"][ex["answer_idx"]]},
            "meta": {"family": ex.get("family")}
        }

def map_3b(path):
    for ex in read_jsonl(path):
        yield {
            "schema_version":"mhbench-v1",
            "id": f"DSM-3B-{ex['id']}",
            "dataset":"DSM","subsource":"3B",
            "split":"TEST",
            "task":"classification","subtype":"symptom2dx",
            "group_id": ex["disorder"],
            "input": {"text": ex["symptoms"]},
            "choices": ex["choices"],
            "target": {"label": ex["choices"][ex["label_idx"]]},
            "meta": {"family": ex.get("family")}
        }

def map_3c(path):
    for ex in read_jsonl(path):
        yield {
            "schema_version":"mhbench-v1",
            "id": f"DSM-3C-{ex['id']}",
            "dataset":"DSM","subsource":"3C",
            "split":"TEST",
            "task":"generation","subtype":"explanation",
            "group_id": ex["disorder"],
            "input": {"question": ex["question"], "context": ex.get("context","")},
            "target": {"rationale": ex.get("gold_points") or ex.get("rationale","")},
            "meta": {"family": ex.get("family")}
        }

def map_3d(path):
    for ex in read_jsonl(path):
        yield {
            "schema_version":"mhbench-v1",
            "id": f"DSM-3D-{ex['id']}",
            "dataset":"DSM","subsource":"3D",
            "split":"TEST",
            "task":"generation","subtype":"vignette",
            "group_id": ex["disorder"],
            "input": {"vignette": ex["vignette"]},
            "target": {"json": ex.get("answers", {})},
            "meta": {"family": ex.get("family")}
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-3a", required=True)
    ap.add_argument("--in-3b", required=True)
    ap.add_argument("--in-3c", required=True)
    ap.add_argument("--in-3d", required=True)
    ap.add_argument("--out", default="mhbench/benchmark/dsm_test.jsonl")
    args = ap.parse_args()

    rows = []
    rows.extend(map_3a(args.in_3a))
    rows.extend(map_3b(args.in_3b))
    rows.extend(map_3c(args.in_3c))
    rows.extend(map_3d(args.in_3d))

    n = write_jsonl(args.out, rows)
    print(f"Wrote DSM TEST: {n} records -> {args.out}")
