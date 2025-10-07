"""
MHBench – Knowledge Metrics
---------------------------
Evaluates factual and definitional knowledge recall for DSM-5 QA tasks.

Metrics:
    • Accuracy (top-1)
    • Top-k accuracy (e.g., top-3)
    • Expected Calibration Error (ECE)
    • Brier Score

Author: Sepehr Ghamari
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score


# -----------------------------------------------------------
# 1. Expected Calibration Error (ECE)
# -----------------------------------------------------------

def expected_calibration_error(y_true, y_pred, confs, bins=10, strategy="uniform"):
    """
    Compute Expected Calibration Error (ECE) and Brier Score.
    """
    confs = np.array([c if (c is not None and 0.0 <= c <= 1.0) else np.nan for c in confs], dtype=float)
    mask = ~np.isnan(confs)
    if mask.sum() == 0:
        return {"ece": None, "brier": None, "valid_samples": 0}

    y_true = np.array(y_true, dtype=object)[mask]
    y_pred = np.array(y_pred, dtype=object)[mask]
    confs = confs[mask]

    correct = (y_true == y_pred).astype(float)
    brier = float(np.mean((correct - confs) ** 2))

    # Bin edges
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(confs, quantiles)
    else:
        bin_edges = np.linspace(0, 1, bins + 1)

    ece = 0.0
    for i in range(bins):
        sel = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
        if sel.sum() == 0:
            continue
        conf_bin = confs[sel].mean()
        acc_bin = correct[sel].mean()
        ece += sel.mean() * abs(acc_bin - conf_bin)

    return {
        "ece": float(ece),
        "brier": float(brier),
        "valid_samples": int(mask.sum())
    }


# -----------------------------------------------------------
# 2. Accuracy and Top-k Accuracy
# -----------------------------------------------------------

def compute_knowledge_accuracy(
    y_true: List[str],
    y_pred: List[str],
    topk_preds: Optional[List[List[str]]] = None,
    k: int = 3
) -> Dict[str, float]:
    """
    Compute accuracy and optional top-k accuracy for DSM-5 QA.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"

    acc = accuracy_score(y_true, y_pred)
    topk_acc = None

    if topk_preds is not None:
        assert all(isinstance(x, list) for x in topk_preds), "topk_preds must be a list of lists"
        correct_topk = sum(true in ranked[:k] for true, ranked in zip(y_true, topk_preds))
        topk_acc = correct_topk / len(y_true)

    return {
        "accuracy": float(acc),
        f"top{k}_accuracy": float(topk_acc) if topk_acc is not None else None
    }


# -----------------------------------------------------------
# 3. Calibration (ECE + Brier)
# -----------------------------------------------------------

def compute_knowledge_calibration(
    y_true: List[str],
    y_pred: List[str],
    confs: Optional[List[float]] = None,
    bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics for QA predictions.
    """
    if confs is None:
        return {"ece": None, "brier": None}
    calib = expected_calibration_error(y_true, y_pred, confs, bins=bins)
    return {"ece": calib["ece"], "brier": calib["brier"]}


# -----------------------------------------------------------
# 4. Combined DSM-QA Knowledge Evaluation
# -----------------------------------------------------------

def evaluate_dsm_qa(
    y_true: List[str],
    y_pred: List[str],
    topk_preds: Optional[List[List[str]]] = None,
    confs: Optional[List[float]] = None,
    k: int = 3
) -> Dict[str, float]:
    """
    Wrapper to compute all knowledge metrics for DSM-5 QA.
    """
    accs = compute_knowledge_accuracy(y_true, y_pred, topk_preds, k=k)
    cals = compute_knowledge_calibration(y_true, y_pred, confs)
    return {**accs, **cals}


# -----------------------------------------------------------
# 5. MHBench compute() Interface
# -----------------------------------------------------------

def compute(instance: Dict, response: Dict):
    """
    Unified MHBench compute() interface for DSM-5 QA knowledge evaluation.

    Args:
        instance (dict): Unified benchmark instance with:
            { "label": str, "options": list[str], "metadata": {...} }
        response (dict): LLM output, e.g.:
            { "label": str, "confidences": [float], "topk": [str] }

    Returns:
        dict: accuracy, top-k accuracy, ECE, Brier
    """
    try:
        y_true = [instance.get("label")]
        y_pred = [response.get("label")]
        confs = response.get("confidences") or [response.get("confidence")]
        topk_preds = [response.get("topk")] if response.get("topk") else None

        metrics = evaluate_dsm_qa(y_true, y_pred, topk_preds, confs, k=3)
        return metrics

    except Exception as e:
        return {
            "accuracy": None,
            "top3_accuracy": None,
            "ece": None,
            "brier": None,
            "error": str(e)
        }


# -----------------------------------------------------------
# 6. Manual Test
# -----------------------------------------------------------

if __name__ == "__main__":
    instance = {"label": "B"}
    response = {
        "label": "B",
        "topk": ["B", "C", "A"],
        "confidence": 0.82
    }

    print("=== MHBench Knowledge Metric Test ===")
    print(compute(instance, response))
