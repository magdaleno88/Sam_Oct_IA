"""Multiclass, binary, and bootstrap evaluation for OCT predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def save_evaluation_artifacts(
    y_true,
    probabilities,
    metrics: dict[str, object],
    output_dir: str | Path,
    class_names=CLASS_NAMES,
) -> dict[str, str]:
    """Save publication-ready metric tables, confusion matrices, and one-vs-rest ROC curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(class_names))
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    summary_rows = [
        {"metric": key, "value": metrics[key]}
        for key in (
            "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "macro_auc", "micro_auc", "cohen_kappa",
        )
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame.from_dict(metrics["per_class"], orient="index").rename_axis(
        "class"
    ).reset_index().to_csv(output_dir / "metrics_per_class.csv", index=False)

    matrix = np.asarray(metrics["confusion_matrix"])
    normalized = np.asarray(metrics["normalized_confusion_matrix"])
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    normalized_frame = pd.DataFrame(normalized, index=class_names, columns=class_names)
    matrix_frame.to_csv(output_dir / "confusion_matrix.csv", index_label="actual")
    normalized_frame.to_csv(
        output_dir / "confusion_matrix_normalized.csv", index_label="actual"
    )

    def plot_matrix(values, title, filename, value_format, colorbar_label):
        fig, axis = plt.subplots(figsize=(7.5, 6.5))
        image = axis.imshow(values, cmap="Blues", vmin=0)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)
        axis.set(
            xticks=labels, yticks=labels, xticklabels=class_names, yticklabels=class_names,
            xlabel="Predicción", ylabel="Clase real", title=title,
        )
        threshold = values.max() / 2 if values.size else 0
        for row in labels:
            for column in labels:
                axis.text(
                    column, row, format(values[row, column], value_format),
                    ha="center", va="center",
                    color="white" if values[row, column] > threshold else "#152238",
                )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    plot_matrix(matrix, "Matriz de confusión — test", "confusion_matrix.png", "d", "Imágenes")
    plot_matrix(
        normalized, "Matriz de confusión normalizada — test",
        "confusion_matrix_normalized.png", ".2f", "Proporción",
    )

    binary = label_binarize(y_true, classes=labels)
    roc_rows = []
    fig, axis = plt.subplots(figsize=(8, 7))
    for index, name in enumerate(class_names):
        if np.unique(binary[:, index]).size < 2:
            continue
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            binary[:, index], probabilities[:, index]
        )
        auc = metrics["per_class"][name]["auc_ovr"]
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"{name} (AUC={auc:.3f})")
        roc_rows.extend(
            {
                "class": name, "false_positive_rate": fpr,
                "true_positive_rate": tpr, "threshold": threshold,
            }
            for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds)
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Azar")
    axis.set(
        xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
        title="Curvas ROC one-vs-rest — test", xlim=(0, 1), ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(roc_rows).to_csv(output_dir / "roc_curves.csv", index=False)

    return {
        path.name: str(path)
        for path in sorted(output_dir.iterdir())
        if path.suffix.lower() in {".csv", ".png"}
    }
