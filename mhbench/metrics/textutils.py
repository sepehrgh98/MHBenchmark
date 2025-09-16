import re, unicodedata
from bs4 import BeautifulSoup

def normalize_text(s: str) -> str:
    if not s: return ""
    s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
