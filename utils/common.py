# utils/common.py
import json, re, unicodedata, hashlib, os, csv, math
from typing import Iterable, Dict, Any, List
import json, re, os, math, unicodedata
from bs4 import BeautifulSoup

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n

URL_RE = re.compile(r'https?://\S+')
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
def canon_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = s.replace("\u200b", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def hash_text_16(s: str) -> str:
    return hashlib.sha256(canon_text(s).encode("utf-8")).hexdigest()[:16]

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def ensure_test_split(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec["split"] = "TEST"
    return rec

def label_from_yesno(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("yes"): return "Depression"
    if s.startswith("no"):  return "NotDepression"
    # allow variants like "Yes." or "Yes – ..." etc.
    if s and s[0] in ("y","n"):
        return "Depression" if s[0]=="y" else "NotDepression"
    return "Unknown"

def extract_rationale(s: str) -> str:
    if not s: return ""
    parts = re.split(r"(?i)reasoning\s*:\s*", s, maxsplit=1)
    return parts[-1].strip() if len(parts)==2 else s.strip()


def strip_html(s: str) -> str:
    if not s: return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)

def normspace(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()

def safe_get(d, *keys, default=None):
    x = d
    for k in keys:
        x = x.get(k, {}) if isinstance(x, dict) else {}
    return x if x else default

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)