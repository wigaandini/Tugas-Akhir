import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

from config import GESTURE_CLASSES, FIGURES_DIR


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def print_report(y_true, y_pred, title=""):
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n{'=' * 50}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"{'=' * 50}")
    print(classification_report(y_true, y_pred, target_names=GESTURE_CLASSES))
    return metrics


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_name=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def measure_latency(predict_fn, X_sample, n_runs=100):
    # Measure average inference time per single window
    # predict_fn: callable that takes a single input and returns prediction
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = np.array(times) * 1000
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def print_latency(latency_dict, model_name=""):
    print(f"\nLatency — {model_name}")
    print(f"  Mean:   {latency_dict['mean_ms']:.2f} ms")
    print(f"  Median: {latency_dict['median_ms']:.2f} ms")
    print(f"  P95:    {latency_dict['p95_ms']:.2f} ms")
    under_300 = "✓" if latency_dict["p95_ms"] < 300 else "✗"
    print(f"  <300ms: {under_300}")
