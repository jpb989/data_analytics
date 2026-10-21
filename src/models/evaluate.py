"""Model evaluation metrics calculation suite.

Computes comprehensive classification metrics:
ROC-AUC, PR-AUC, Macro-F1, Weighted-F1, Brier Score, Precision, Recall, Accuracy.
"""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted class labels (0 or 1).
        y_proba: Predicted probabilities for the positive class (class 1).

    Returns:
        Dictionary containing all evaluation metrics rounded to 4 decimals.
    """
    # Defensive clipping on probabilities
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float(np.mean(y_true))

    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except Exception:
        brier = 1.0

    try:
        loss = float(log_loss(y_true, y_proba))
    except Exception:
        loss = 999.0

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_binary": round(f1_binary, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
    }


def compute_metric_degradation(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    primary_metric: str = "f1_macro",
) -> Dict[str, Any]:
    """Compute absolute and percentage degradation from baseline for each metric."""
    degradations = {}
    for metric_name, base_val in baseline_metrics.items():
        if metric_name in current_metrics:
            curr_val = current_metrics[metric_name]
            abs_drop = base_val - curr_val
            pct_drop = (abs_drop / base_val * 100.0) if base_val != 0 else 0.0
            degradations[metric_name] = {
                "baseline": base_val,
                "current": curr_val,
                "absolute_drop": round(abs_drop, 4),
                "relative_pct_drop": round(pct_drop, 2),
            }

    primary_abs = degradations.get(primary_metric, {}).get("absolute_drop", 0.0)
    primary_pct = degradations.get(primary_metric, {}).get("relative_pct_drop", 0.0)

    return {
        "primary_metric": primary_metric,
        "primary_absolute_drop": primary_abs,
        "primary_relative_pct_drop": primary_pct,
        "metrics": degradations,
    }
