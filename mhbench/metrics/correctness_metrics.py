"""
MHBench – Correctness Metrics
-----------------------------
Measures predictive correctness for classification or QA tasks.

Implements:
    • compute_cls_metrics
        - Accuracy
        - Macro / Weighted F1
        - Confusion Matrix
Author: Sepehr Ghamari
"""

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================================================
# Core computation function
# =========================================================
def compute_cls_metrics(y_true, y_pred, labels=None, return_cm=True, average_f1=("macro", "weighted")):
    """
    Compute correctness metrics for classification-style tasks.

    Args:
        y_true (list[str] | list[int]): Ground-truth labels.
        y_pred (list[str] | list[int]): Predicted labels.
        labels (list, optional): Label order to enforce in metrics.
        return_cm (bool): Whether to include confusion matrix in output.
        average_f1 (tuple): Which F1 averaging modes to compute.

    Returns:
        dict: {
            "accuracy": float,
            "f1_macro": float,
            "f1_weighted": float,
            "confusion_matrix": list[list[int]] | None,
            "labels": list[str] | None
        }
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length."

    acc = accuracy_score(y_true, y_pred)
    out = {"accuracy": float(acc)}

    for mode in average_f1:
        out[f"f1_{mode}"] = float(f1_score(y_true, y_pred, average=mode, labels=labels))

    if return_cm:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["confusion_matrix"] = cm.tolist()
        out["labels"] = list(labels) if labels is not None else None
    else:
        out["confusion_matrix"] = None
        out["labels"] = None

    return out


# =========================================================
# Unified MHBenchmark interface
# =========================================================
def compute(instance, response):
    """
    MHBenchmark compute() entrypoint.
    Extracts true/predicted labels for classification or QA tasks.

    Args:
        instance (dict): Unified benchmark instance (with `reference.label`).
        response (dict): Model's response (with `label`).

    Returns:
        dict: Correctness metrics (accuracy, F1s, confusion matrix)
    """
    try:
        # Pull labels
        y_true = [instance.get("reference", {}).get("label")]
        y_pred = [response.get("label")]

        # Optionally, allow label list in metadata for fixed label order
        labels = instance.get("metadata", {}).get("label_set", None)

        result = compute_cls_metrics(y_true, y_pred, labels=labels, return_cm=False)
        return result

    except Exception as e:
        return {"accuracy": None, "f1_macro": None, "f1_weighted": None, "error": str(e)}


# =========================================================
# Manual test
# =========================================================
if __name__ == "__main__":
    instance = {"reference": {"label": "support"}}
    response = {"label": "support"}
    print(compute(instance, response))
