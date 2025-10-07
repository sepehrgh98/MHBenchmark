"""
MHBench – Calibration Metrics
-----------------------------
Measures model confidence reliability.

Implements:
    • Expected Calibration Error (ECE)
    • Brier Score
Author: Sepehr Ghamari
"""

import numpy as np


# =========================================================
# Core computation function
# =========================================================
def expected_calibration_error(y_true, y_pred, confs, bins=10, strategy="uniform"):
    """
    Compute Expected Calibration Error (ECE) and Brier Score.

    Args:
        y_true (list[str] | list[int]): True labels.
        y_pred (list[str] | list[int]): Predicted labels.
        confs (list[float]): Confidence for each prediction (probability of predicted class).
        bins (int): Number of bins for ECE computation.
        strategy (str): 'uniform' or 'quantile' binning.

    Returns:
        dict: {"ece": float, "brier": float, "valid_samples": int}
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


# =========================================================
# Unified MHBenchmark interface
# =========================================================
def compute(instance, response):
    """
    MHBenchmark compute() entrypoint.
    Extracts y_true, y_pred, and confidence for calibration analysis.

    Args:
        instance (dict): Unified benchmark instance.
        response (dict): Model's JSON output.

    Returns:
        dict: {"ece": float, "brier": float, "valid_samples": int}
    """
    try:
        # Expected structure:
        # instance["reference"]["label"] -> true label
        # response["label"] -> predicted label
        # response["confidence"] -> model confidence (0.0–1.0)
        y_true = [instance.get("reference", {}).get("label")]
        y_pred = [response.get("label")]
        confs = [response.get("confidence", None)]

        # Some models might return nested confidence (e.g., {"confidence": {"value": 0.9}})
        if isinstance(confs[0], dict) and "value" in confs[0]:
            confs[0] = confs[0]["value"]

        result = expected_calibration_error(y_true, y_pred, confs, bins=10, strategy="uniform")
        return result

    except Exception as e:
        return {"ece": None, "brier": None, "valid_samples": 0, "error": str(e)}


# =========================================================
# Manual test
# =========================================================
if __name__ == "__main__":
    # Dummy example
    instance = {"reference": {"label": "support"}}
    response = {"label": "support", "confidence": 0.9}
    print(compute(instance, response))
