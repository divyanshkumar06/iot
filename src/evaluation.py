"""
src/evaluation.py — Model Evaluation Utilities
===============================================
Computes and formats all performance metrics for both models.

Metrics
-------
- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-score (macro average)
- Confusion matrix
- Simulated inference latency

Author (proposed): Vishal
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.neural_network import MLPClassifier


def evaluate_model(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    n_latency_runs: int = 100,
) -> dict:
    """
    Compute full evaluation metrics for a trained model.

    Parameters
    ----------
    model          : trained sklearn MLPClassifier
    X_test         : scaled feature array
    y_test         : true labels
    model_name     : label used in reports
    n_latency_runs : number of inference calls for latency estimation

    Returns
    -------
    dict with all metrics
    """
    y_pred = model.predict(X_test)

    # ── Classification metrics ────────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred,
                                      target_names=["Normal", "Fault"],
                                      zero_division=0)

    # ── Simulated inference latency ───────────────────────────────────────────
    # Measure time to classify a single sample, repeated n_latency_runs times
    single_sample = X_test[:1]
    latency_times = []
    for _ in range(n_latency_runs):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        latency_times.append(time.perf_counter() - t0)

    # Remove first few (JIT warm-up) and take mean
    latency_ms = np.mean(latency_times[5:]) * 1000.0

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference_latency_ms": round(latency_ms, 4),
        "n_test_samples": len(y_test),
        "y_pred": y_pred.tolist(),
        "y_true": y_test.tolist(),
    }


def compare_models(fed_metrics: dict, central_metrics: dict) -> dict:
    """
    Build a side-by-side comparison dict.
    """
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score",
                          "inference_latency_ms"]
    comparison = {}
    for metric in metrics_to_compare:
        fed_val = fed_metrics.get(metric, 0)
        cen_val = central_metrics.get(metric, 0)
        comparison[metric] = {
            "federated": fed_val,
            "centralized": cen_val,
            "difference": round(fed_val - cen_val, 4),
        }
    return comparison


def print_comparison_table(fed_metrics: dict, central_metrics: dict) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 65)
    print(f"{'METRIC':<28} {'FEDERATED':>12} {'CENTRALIZED':>12} {'DIFF':>8}")
    print("=" * 65)
    for key in ["accuracy", "precision", "recall", "f1_score", "inference_latency_ms"]:
        fed_v = fed_metrics.get(key, 0)
        cen_v = central_metrics.get(key, 0)
        unit = " ms" if "latency" in key else ""
        print(f"  {key:<26} {fed_v:>11.4f} {cen_v:>11.4f} {fed_v-cen_v:>+8.4f}{unit}")
    print("=" * 65)

