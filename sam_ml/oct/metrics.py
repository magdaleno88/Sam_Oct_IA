"""Multiclass, binary, and bootstrap evaluation for OCT predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from sam_ml.oct.constants import CLASS_NAMES


def specificity_per_class(y_true, y_pred, num_classes: int = 4) -> list[float]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = matrix.sum()
    values = []
    for index in range(num_classes):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        tn = total - tp - fp - fn
        values.append(float(tn / (tn + fp)) if tn + fp else float("nan"))
    return values


def _safe_auc(y_binary: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_binary, probabilities)) if np.unique(y_binary).size == 2 else float("nan")


def evaluate_predictions(y_true, probabilities, class_names=CLASS_NAMES) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    binary = label_binarize(y_true, classes=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    specificity = specificity_per_class(y_true, y_pred, len(class_names))
    aucs = [_safe_auc(binary[:, i], probabilities[:, i]) for i in labels]
    average_precision = [
        float(average_precision_score(binary[:, i], probabilities[:, i]))
        if binary[:, i].sum() else float("nan") for i in labels
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    abnormal_true = (y_true != class_names.index("NORMAL")).astype(int)
    abnormal_probability = 1 - probabilities[:, class_names.index("NORMAL")]
    abnormal_pred = (abnormal_probability >= 0.5).astype(int)
    abnormal_cm = confusion_matrix(abnormal_true, abnormal_pred, labels=[0, 1])
    tn, fp, fn, tp = abnormal_cm.ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall_sensitivity": float(recall[i]),
                   "specificity": specificity[i], "f1": float(f1[i]), "support": int(support[i]),
                   "auc_ovr": aucs[i], "average_precision": average_precision[i]}
            for i, name in enumerate(class_names)
        },
        "macro_auc": float(np.nanmean(aucs)),
        "micro_auc": _safe_auc(binary.ravel(), probabilities.ravel()),
        "abnormal_vs_normal": {
            "accuracy": float(accuracy_score(abnormal_true, abnormal_pred)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "roc_auc": _safe_auc(abnormal_true, abnormal_probability),
        },
    }
    result["binary_vs_normal"] = {}
    normal = class_names.index("NORMAL")
    for disease in range(normal):
        mask = np.isin(y_true, [disease, normal])
        truth = (y_true[mask] == disease).astype(int)
        conditional = probabilities[mask, disease] / np.maximum(
            probabilities[mask, disease] + probabilities[mask, normal], 1e-12
        )
        pred = (conditional >= 0.5).astype(int)
        binary_cm = confusion_matrix(truth, pred, labels=[0, 1])
        tn, fp, fn, tp = binary_cm.ravel()
        result["binary_vs_normal"][f"{class_names[disease]}_vs_NORMAL"] = {
            "accuracy": float(accuracy_score(truth, pred)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "roc_auc": _safe_auc(truth, conditional),
        }
    return result


def bootstrap_confidence_intervals(y_true, probabilities, iterations: int = 1000, seed: int = 42):
    """Image-level percentile CIs; use grouped resampling externally when patient IDs exist."""
    rng = np.random.default_rng(seed)
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    values = {"accuracy": [], "macro_f1": [], "macro_auc": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        try:
            metrics = evaluate_predictions(y_true[indices], probabilities[indices])
            for key in values:
                values[key].append(metrics[key])
        except ValueError:
            continue
    return {key: {"lower": float(np.nanpercentile(items, 2.5)),
                  "upper": float(np.nanpercentile(items, 97.5))} for key, items in values.items()}


def save_metrics(metrics: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
