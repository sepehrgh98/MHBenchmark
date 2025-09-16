# scripts/validate.py
import sys, json
from jsonschema import Draft202012Validator
from utils.common import read_jsonl

schema = json.load(open("schemas/canonical.schema.json","r",encoding="utf-8"))
validator = Draft202012Validator(schema)

def validate_file(path: str) -> int:
    errs = 0
    for i, rec in enumerate(read_jsonl(path), start=1):
        for e in validator.iter_errors(rec):
            print(f"[{path}:{i}] {e.message}")
            errs += 1
    return errs

if __name__ == "__main__":
    total = 0
    for p in sys.argv[1:]:
        total += validate_file(p)
    if total:
        print(f"Validation FAILED with {total} error(s)."); sys.exit(1)
    print("Validation OK.")
