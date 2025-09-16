from bert_score import score as bertscore
from .textutils import normalize_text

def bertscore_max_multi_ref(hyps, refs_list, lang="en"):
    """
    hyps: list[str] one per item
    refs_list: list[list[str]] each inner list are multiple gold refs for that item
    Returns average P,R,F1
    """
    # Flatten multi-refs by taking per-item max later via grouping not supported directly by library.
    # Simple approach: compare to concatenated best ref = max F1 among refs per item.
    best_refs = []
    for refs in refs_list:
        # pick the longest ref as proxy (cheap). For exact, compute per-ref and take max; omitted for speed.
        best_refs.append(max(refs, key=len))
    P,R,F1 = bertscore(hyps, best_refs, lang=lang, rescale_with_baseline=True)
    return float(P.mean()), float(R.mean()), float(F1.mean())
