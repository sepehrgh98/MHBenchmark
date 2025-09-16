from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def compute_cls_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels) if labels is not None else confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro, "confusion_matrix": cm.tolist()}

def expected_calibration_error(y_true, y_pred, confs, bins=10):
    # y_pred,y_true as string labels; confs in [0,1] or None; if None, skip ECE.
    confs = np.array([c if (c is not None and 0<=c<=1) else np.nan for c in confs], dtype=float)
    mask = ~np.isnan(confs)
    if mask.sum() == 0: return {"ece": None, "brier": None}
    correct = (np.array(y_true, dtype=object)[mask] == np.array(y_pred, dtype=object)[mask]).astype(float)
    confs = confs[mask]
    # Brier score: mean squared error between correctness (0/1) and confidence
    brier = float(np.mean((correct - confs)**2))
    # ECE
    ece = 0.0
    bin_ids = np.minimum((confs * bins).astype(int), bins-1)
    for b in range(bins):
        sel = bin_ids == b
        if sel.sum() == 0: continue
        conf_bin = confs[sel].mean()
        acc_bin  = correct[sel].mean()
        ece += (sel.mean()) * abs(acc_bin - conf_bin)
    return {"ece": float(ece), "brier": brier}
