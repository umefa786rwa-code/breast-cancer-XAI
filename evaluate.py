"""
evaluate.py
===========
Model evaluation with all metrics from thesis Tables 3.4, 3.5, 3.6.

Computes:
    - Accuracy, Precision, Sensitivity (Recall), Specificity
    - F1 Score, MCC (Matthews Correlation Coefficient)
    - Confusion matrix
    - ROC curve and AUC

Thesis results (for reference):
    CNN:      Accuracy 96%, Precision 95%, MCC 91%
    ResNet50: Accuracy 97%, Precision 98%, MCC 95%
    VGG-16:   Accuracy 98%, Precision 99%, MCC 97%  ← BEST

Usage:
    python evaluate.py --model vgg16
    python evaluate.py --all
    python evaluate.py --compare  # Side-by-side comparison table

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, auc, classification_report
)

from preprocessing import load_mias_dataset
from sklearn.model_selection import train_test_split


# ── Configuration ─────────────────────────────────────────────────────────────
SAVE_DIR     = "saved_models/"
FIGURES_DIR  = "figures/"
TEST_SIZE    = 0.25
RANDOM_SEED  = 42
CLASS_NAMES  = ["Benign", "Malignant"]

os.makedirs(FIGURES_DIR, exist_ok=True)

# Thesis results for comparison (Tables 3.4, 3.5, 3.6)
THESIS_RESULTS = {
    "CNN": {
        "Accuracy": 0.96, "Precision": 0.95,
        "Sensitivity": 0.97, "Specificity": 0.95,
        "MCC": 0.91, "F1": 0.96
    },
    "ResNet-50": {
        "Accuracy": 0.97, "Precision": 0.98,
        "Sensitivity": 0.97, "Specificity": 0.98,
        "MCC": 0.95, "F1": 0.97
    },
    "VGG-16": {
        "Accuracy": 0.98, "Precision": 0.99,
        "Sensitivity": 0.98, "Specificity": 0.99,
        "MCC": 0.97, "F1": 0.98
    }
}


# ── Metrics computation ────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics from thesis Tables 3.4-3.6.
    
    Formulas (thesis Section 3.9):
        Accuracy    = (TP + TN) / (TP + TN + FP + FN)
        Sensitivity = TP / (TP + FN)              [Recall]
        Specificity = TN / (TN + FP)
        Precision   = TP / (TP + FP)
        F1 Score    = 2 * Precision * Recall / (Precision + Recall)
        MCC         = (TP*TN - FP*FN) / 
                      sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Args:
        y_true: Ground truth labels (0=Benign, 1=Malignant)
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metric name → value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    mcc         = matthews_corrcoef(y_true, y_pred)
    
    return {
        "Accuracy":    accuracy,
        "Precision":   precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score":    f1,
        "MCC":         mcc,
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn)
    }


def print_metrics_table(metrics: dict, model_name: str):
    """
    Print evaluation metrics in thesis table format.
    
    Args:
        metrics:    Dict from compute_metrics()
        model_name: Model name string for display
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results — {model_name.upper()}")
    print(f"(Thesis reference: Tables 3.4, 3.5, 3.6)")
    print('='*60)
    
    metric_keys = [
        "Accuracy", "Precision", "Sensitivity",
        "Specificity", "F1 Score", "MCC"
    ]
    
    for key in metric_keys:
        val = metrics[key]
        pct = f"{val*100:.1f}%"
        bar = "█" * int(val * 20)
        print(f"  {key:<14} {pct:>6}  {bar}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['TP']}  FP={metrics['FP']}")
    print(f"    FN={metrics['FN']}  TN={metrics['TN']}")
    print('='*60)


# ── Confusion matrix plot ──────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None
):
    """
    Plot confusion matrix heatmap.
    
    Reproduces thesis Figure 4 (confusion matrix).
    
    Args:
        y_true:     Ground truth labels
        y_pred:     Predicted labels
        model_name: Model name for title
        save_path:  Optional save path
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1, linecolor="white",
        annot_kws={"size": 16, "weight": "bold"}
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(
        f"Confusion Matrix — {model_name.upper()}\n"
        f"(Accuracy: {accuracy_score(y_true, y_pred)*100:.1f}%)",
        fontsize=12, fontweight="bold", pad=15
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


# ── ROC curve ─────────────────────────────────────────────────────────────────
def plot_roc_curve(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    save_path: str = None
):
    """
    Plot ROC curve and compute AUC.
    
    Reproduces thesis Figure 5 (ROC curve on CBIS-DDSM
    cross-dataset validation).
    
    Args:
        model:      Trained Keras model
        X_test:     Test images
        y_test:     True labels
        model_name: Model name for legend
        save_path:  Optional save path
    """
    # Get prediction probabilities for malignant class
    y_prob = model.predict(X_test, verbose=0)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.plot(
        fpr, tpr,
        color="#2A2460", linewidth=2.5,
        label=f"{model_name.upper()} (AUC = {roc_auc:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1],
        color="#AAAAAA", linewidth=1.5,
        linestyle="--", label="Random Classifier (AUC = 0.500)"
    )
    
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2A2460")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(
        f"ROC Curve — {model_name.upper()}\n"
        f"MIAS Mammography Dataset",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()
    return roc_auc


# ── Model comparison ───────────────────────────────────────────────────────────
def compare_all_models(results: dict, save_path: str = None):
    """
    Create side-by-side comparison bar chart for all three models.
    
    Reproduces the comparison implied by thesis Tables 3.4-3.6.
    
    Args:
        results:   Dict of {model_name: metrics_dict}
        save_path: Optional save path
    """
    metrics_to_plot = [
        "Accuracy", "Precision", "Sensitivity",
        "Specificity", "F1 Score", "MCC"
    ]
    
    model_names = list(results.keys())
    colors = ["#0D6E5C", "#2A2460", "#C0392B"]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name].get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(
            x + i * width, vals,
            width, label=name.upper(),
            color=color, alpha=0.85,
            edgecolor="white", linewidth=0.8
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val*100:.0f}%",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )
    
    ax.set_xlabel("Metric", fontsize=12, labelpad=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Model Performance Comparison — MIAS Mammography Dataset\n"
        "CNN vs ResNet-50 vs VGG-16 (Thesis Tables 3.4, 3.5, 3.6)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylim([0.85, 1.02])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved to: {save_path}")
    
    plt.show()


def print_comparison_table(results: dict):
    """
    Print thesis-style comparison table for all models.
    
    Args:
        results: Dict of {model_name: metrics_dict}
    """
    metrics = [
        "Accuracy", "Precision", "Sensitivity",
        "Specificity", "F1 Score", "MCC"
    ]
    
    print("\n" + "="*70)
    print("MODEL COMPARISON — MIAS MAMMOGRAPHY DATASET")
    print("(Thesis Tables 3.4, 3.5, 3.6)")
    print("="*70)
    
    # Header
    header = f"{'Metric':<16}"
    for name in results:
        header += f"{name:>14}"
    print(header)
    print("-"*70)
    
    # Rows
    for m in metrics:
        row = f"{m:<16}"
        vals = [results[name].get(m, 0) for name in results]
        best_idx = np.argmax(vals)
        for i, val in enumerate(vals):
            cell = f"{val*100:.1f}%"
            if i == best_idx:
                cell += " ←"
            row += f"{cell:>14}"
        print(row)
    
    print("="*70)
    print("← indicates best performing model for each metric")
    print("\nThesis conclusion: VGG-16 is the best model overall")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained breast cancer classification models"
    )
    parser.add_argument(
        "--model", type=str, default="vgg16",
        choices=["cnn", "resnet50", "vgg16"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all three models"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Show comparison table using thesis results"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/mias/",
        help="Path to MIAS dataset"
    )
    args = parser.parse_args()
    
    # Quick comparison using thesis results (no model needed)
    if args.compare:
        print_comparison_table(THESIS_RESULTS)
        compare_all_models(
            THESIS_RESULTS,
            save_path=os.path.join(FIGURES_DIR, "model_comparison.png")
        )
        return
    
    # Load test data
    print("Loading test data...")
    X, y, _ = load_mias_dataset(args.data_dir, mode="binary")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y
    )
    
    # Models to evaluate
    model_names = (
        ["cnn", "resnet50", "vgg16"] if args.all else [args.model]
    )
    
    all_results = {}
    
    for name in model_names:
        model_path = os.path.join(SAVE_DIR, f"{name}_best.h5")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print(f"Run: python train.py --model {name}")
            continue
        
        print(f"\nLoading {name.upper()} from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Predictions
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        # Compute and display metrics
        metrics = compute_metrics(y_test, y_pred)
        print_metrics_table(metrics, name)
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES
        ))
        
        # Plots
        plot_confusion_matrix(
            y_test, y_pred, name,
            save_path=os.path.join(FIGURES_DIR, f"confusion_{name}.png")
        )
        
        roc_auc = plot_roc_curve(
            model, X_test, y_test, name,
            save_path=os.path.join(FIGURES_DIR, f"roc_{name}.png")
        )
        
        metrics["AUC"] = roc_auc
        all_results[name.upper()] = metrics
    
    # If multiple models evaluated, show comparison
    if len(all_results) > 1:
        print_comparison_table(all_results)
        compare_all_models(
            all_results,
            save_path=os.path.join(FIGURES_DIR, "model_comparison.png")
        )


if __name__ == "__main__":
    main()
