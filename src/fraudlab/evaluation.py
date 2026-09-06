from __future__ import print_function

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_BUDGET_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def choose_threshold(y_true, scores):
    """Choose a decision threshold on validation data by binary F1."""
    candidates = np.unique(np.percentile(scores, np.linspace(0, 100, 201)))
    best = (0.5, -1.0)
    for threshold in candidates:
        value = f1_score(y_true, scores >= threshold, zero_division=0)
        if value > best[1]:
            best = (float(threshold), float(value))
    return best[0]


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _calibration_errors(y_true, scores, n_bins=10):
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(y_true[mask])) - float(np.mean(scores[mask])))
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _ranking_metrics(y_true, scores, positive_count, budget_fractions):
    order = np.argsort(scores, kind="stable")[::-1]
    prevalence = _safe_divide(positive_count, len(y_true))
    result = {}

    def add_at_k(prefix, k):
        k = min(max(1, int(k)), len(y_true))
        hits = int(np.sum(y_true[order[:k]]))
        precision = _safe_divide(hits, k)
        recall = _safe_divide(hits, positive_count)
        result[f"precision_at_{prefix}"] = precision
        result[f"recall_at_{prefix}"] = recall
        result[f"f1_at_{prefix}"] = _safe_divide(2.0 * precision * recall, precision + recall)
        result[f"lift_at_{prefix}"] = _safe_divide(precision, prevalence)
        result[f"hits_at_{prefix}"] = hits
        result[f"k_at_{prefix}"] = k

    add_at_k("k_equals_test_positives", positive_count)
    for fraction in budget_fractions:
        label = f"top_{fraction * 100:g}pct".replace(".", "p")
        add_at_k(label, int(np.ceil(len(y_true) * fraction)))
    return result


def _operating_point_metrics(y_true, scores):
    result = {}
    if len(np.unique(y_true)) < 2:
        return result
    fpr, tpr, _ = roc_curve(y_true, scores)
    for target in (0.001, 0.005, 0.01, 0.05, 0.10):
        feasible = tpr[fpr <= target]
        label = f"{target * 100:g}pct".replace(".", "p")
        result[f"recall_at_fpr_{label}"] = float(np.max(feasible)) if len(feasible) else 0.0
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    for target in (0.50, 0.70, 0.80, 0.90):
        feasible = precision_curve[recall_curve >= target]
        label = f"{target * 100:g}pct".replace(".", "p")
        result[f"max_precision_at_recall_ge_{label}"] = float(np.max(feasible)) if len(feasible) else 0.0
    return result


def evaluate(y_true, scores, threshold, budget_fractions=DEFAULT_BUDGET_FRACTIONS):
    """Return a broad, threshold-explicit fraud-detection metric superset.

    Existing result keys are preserved. ``pr_auc`` remains sklearn Average
    Precision for backward compatibility; ``pr_auc_trapezoidal`` records the
    trapezoidal area under the precision-recall curve explicitly.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty test set")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinity")

    pred = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    positive_count = int(np.sum(y_true))
    negative_count = int(len(y_true) - positive_count)
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    precision_normal = float(precision_score(y_true, pred, pos_label=0, zero_division=0))
    recall_normal = float(recall_score(y_true, pred, pos_label=0, zero_division=0))
    f1_normal = float(f1_score(y_true, pred, pos_label=0, zero_division=0))
    f1_fraud = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
    specificity = _safe_divide(tn, negative_count)
    npv = _safe_divide(tn, tn + fn)
    fpr_value = _safe_divide(fp, negative_count)
    fnr_value = _safe_divide(fn, positive_count)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, scores)
    average_precision = float(average_precision_score(y_true, scores))

    result = {
        "threshold": float(threshold),
        "threshold_selection": "validation_binary_f1",
        "pr_auc": average_precision,
        "average_precision": average_precision,
        "pr_auc_trapezoidal": float(auc(pr_recall[::-1], pr_precision[::-1])),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1_fraud,
        "precision_normal": precision_normal,
        "recall_normal": recall_normal,
        "f1_normal": f1_normal,
        "precision_fraud": precision,
        "recall_fraud": recall,
        "f1_fraud": f1_fraud,
        "f0_5": float(fbeta_score(y_true, pred, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "specificity": specificity,
        "negative_predictive_value": npv,
        "false_positive_rate": fpr_value,
        "false_negative_rate": fnr_value,
        "false_discovery_rate": _safe_divide(fp, tp + fp),
        "false_omission_rate": _safe_divide(fn, tn + fn),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "g_mean": float(np.sqrt(max(0.0, recall * specificity))),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "prevalence": _safe_divide(positive_count, len(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "test_rows": int(len(y_true)),
        "test_fraud": positive_count,
        "test_non_fraud": negative_count,
    }

    ranking = _ranking_metrics(y_true, scores, positive_count, budget_fractions)
    result.update(ranking)
    result["precision_at_k_equals_test_positives"] = ranking["precision_at_k_equals_test_positives"]
    result.update(_operating_point_metrics(y_true, scores))

    try:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        result["roc_auc"] = None

    if np.all((scores >= 0.0) & (scores <= 1.0)):
        clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
        result["brier_score"] = float(brier_score_loss(y_true, scores))
        result["log_loss"] = float(log_loss(y_true, np.column_stack([1.0 - clipped, clipped]), labels=[0, 1]))
        ece, mce = _calibration_errors(y_true, scores, n_bins=10)
        result["expected_calibration_error_10_bin"] = ece
        result["maximum_calibration_error_10_bin"] = mce
    else:
        result["brier_score"] = None
        result["log_loss"] = None
        result["expected_calibration_error_10_bin"] = None
        result["maximum_calibration_error_10_bin"] = None
    return result


def save_prediction_artifact(path, y_true, scores, threshold, test_indices=None):
    """Persist compact test predictions so metrics can be recomputed later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": np.asarray(y_true, dtype=np.int8).reshape(-1),
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
        "threshold": np.asarray([threshold], dtype=np.float64),
    }
    if test_indices is not None:
        payload["test_indices"] = np.asarray(test_indices, dtype=np.int64).reshape(-1)
    np.savez_compressed(path, **payload)
    return path
