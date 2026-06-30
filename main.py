"""
main.py
=======
Complete pipeline for reproducing thesis results.

Runs the full workflow:
    1. Preprocess MIAS dataset
    2. Train CNN, ResNet-50, VGG-16
    3. Evaluate all models
    4. Generate Grad-CAM visualisations
    5. Reproduce survey analysis figures

Usage:
    python main.py --all              # Full pipeline
    python main.py --eval_only        # Evaluate only (models already trained)
    python main.py --survey_only      # Survey figures only
    python main.py --gradcam_only     # Grad-CAM only

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf

from preprocessing import load_mias_dataset, IMAGE_SIZE
from augmentation import augment_batch
from models import get_model
from train import load_and_split_data, train_model, plot_training_history
from evaluate import (
    compute_metrics, print_metrics_table,
    plot_confusion_matrix, plot_roc_curve,
    compare_all_models, print_comparison_table,
    THESIS_RESULTS, FIGURES_DIR, SAVE_DIR
)
from gradcam_visualization import GradCAM, visualise_gradcam, visualise_batch
from survey_analysis import (
    print_survey_summary,
    plot_survey_responses,
    plot_clinical_trust_gap
)
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.25
DATA_DIR    = "data/mias/"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,    exist_ok=True)


# ── Pipeline steps ─────────────────────────────────────────────────────────────
def step1_load_data():
    """Load and split MIAS dataset."""
    print("\n" + "="*60)
    print("STEP 1: Loading and preprocessing MIAS dataset")
    print("="*60)
    return load_and_split_data(DATA_DIR)


def step2_train_all(X_train, X_test, y_train, y_test):
    """Train all three models."""
    print("\n" + "="*60)
    print("STEP 2: Training CNN, ResNet-50, VGG-16")
    print("="*60)
    
    histories = {}
    for name in ["cnn", "resnet50", "vgg16"]:
        histories[name] = train_model(
            name, X_train, X_test, y_train, y_test
        )
    
    plot_training_history(
        histories,
        save_path=os.path.join(FIGURES_DIR, "training_curves_all.png")
    )
    return histories


def step3_evaluate_all(X_test, y_test):
    """Evaluate all three trained models."""
    print("\n" + "="*60)
    print("STEP 3: Evaluating all models")
    print("="*60)
    
    all_results = {}
    
    for name in ["cnn", "resnet50", "vgg16"]:
        model_path = os.path.join(SAVE_DIR, f"{name}_best.h5")
        if not os.path.exists(model_path):
            print(f"Skipping {name} — model not found at {model_path}")
            continue
        
        model   = tf.keras.models.load_model(model_path)
        y_prob  = model.predict(X_test, verbose=0)
        y_pred  = np.argmax(y_prob, axis=1)
        metrics = compute_metrics(y_test, y_pred)
        
        print_metrics_table(metrics, name)
        
        plot_confusion_matrix(
            y_test, y_pred, name,
            save_path=os.path.join(FIGURES_DIR, f"confusion_{name}.png")
        )
        plot_roc_curve(
            model, X_test, y_test, name,
            save_path=os.path.join(FIGURES_DIR, f"roc_{name}.png")
        )
        all_results[name.upper()] = metrics
    
    if len(all_results) > 1:
        print_comparison_table(all_results)
        compare_all_models(
            all_results,
            save_path=os.path.join(FIGURES_DIR, "model_comparison.png")
        )
    
    return all_results


def step4_gradcam(X_test, y_test):
    """Generate Grad-CAM heat maps using best model (VGG-16)."""
    print("\n" + "="*60)
    print("STEP 4: Generating Grad-CAM visualisations (VGG-16)")
    print("="*60)
    
    model_path = os.path.join(SAVE_DIR, "vgg16_best.h5")
    if not os.path.exists(model_path):
        print(f"VGG-16 model not found at {model_path}")
        print("Run step 2 first to train the models.")
        return
    
    model = tf.keras.models.load_model(model_path)
    gcam  = GradCAM(model)
    
    print(f"\nGenerating Grad-CAM for {min(8, len(X_test))} test images...")
    
    fig, axes = plt.subplots(
        min(8, len(X_test)), 3,
        figsize=(15, 4 * min(8, len(X_test)))
    )
    
    import cv2, matplotlib.pyplot as plt
    
    class_names = ["Benign", "Malignant"]
    colors_map  = ["#2A7A2A", "#C0392B"]
    
    for i in range(min(8, len(X_test))):
        img       = X_test[i]
        true_label = y_test[i]
        
        pred      = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        pred_class = np.argmax(pred)
        confidence = pred[pred_class] * 100
        
        heatmap = gcam.compute_heatmap(img, pred_class)
        
        # Convert preprocessed img back to uint8 for display
        img_uint8 = np.uint8(255 * img)
        heatmap_col, superimposed = gcam.overlay_heatmap(heatmap, img_uint8)
        
        row = axes[i] if min(8, len(X_test)) > 1 else axes
        row[0].imshow(img_uint8)
        row[0].set_title(
            f"True: {class_names[true_label]}",
            fontsize=9
        )
        row[0].axis("off")
        
        row[1].imshow(heatmap_col)
        row[1].set_title("Grad-CAM Heat Map", fontsize=9)
        row[1].axis("off")
        
        row[2].imshow(superimposed)
        row[2].set_title(
            f"Pred: {class_names[pred_class]} ({confidence:.0f}%)",
            fontsize=9,
            color=colors_map[pred_class]
        )
        row[2].axis("off")
    
    if min(8, len(X_test)) > 1:
        for ax, title in zip(
            axes[0],
            ["Original", "Grad-CAM Heat Map", "Overlay + Prediction"]
        ):
            ax.set_title(title, fontsize=11, fontweight="bold",
                         color="#2A2460", pad=10)
    
    plt.suptitle(
        "Grad-CAM Visualisations — VGG-16 on MIAS Mammography\n"
        "These outputs were shown to radiologists. "
        "Only 30% correctly identified the disease.",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, "gradcam_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Grad-CAM results saved to: {save_path}")
    plt.show()


def step5_survey():
    """Generate all survey analysis figures."""
    print("\n" + "="*60)
    print("STEP 5: Survey analysis and clinical trust gap visualisation")
    print("="*60)
    
    print_survey_summary()
    plot_survey_responses(
        save_path=os.path.join(FIGURES_DIR, "survey_responses.png")
    )
    plot_clinical_trust_gap(
        save_path=os.path.join(FIGURES_DIR, "clinical_trust_gap.png")
    )


# ── Quick demo (no MIAS dataset needed) ───────────────────────────────────────
def run_demo():
    """
    Demo mode — runs survey analysis and shows thesis results
    without requiring the MIAS dataset download.
    
    Perfect for quickly verifying the repository works.
    """
    print("\n" + "="*60)
    print("DEMO MODE — No dataset required")
    print("Reproducing thesis survey results and comparison tables")
    print("="*60)
    
    # Survey figures (no data needed)
    step5_survey()
    
    # Model comparison from thesis tables
    print("\nThesis model comparison (Tables 3.4, 3.5, 3.6):")
    print_comparison_table(THESIS_RESULTS)
    compare_all_models(
        THESIS_RESULTS,
        save_path=os.path.join(FIGURES_DIR, "thesis_comparison.png")
    )
    
    print("\n" + "="*60)
    print("Demo complete. Figures saved in: figures/")
    print("\nTo run full pipeline with MIAS dataset:")
    print("  1. Download MIAS from: https://www.kaggle.com/datasets/kmader/mias-mammography")
    print("  2. Organise as: data/mias/BENIGN/ and data/mias/MALIGNANT/")
    print("  3. Run: python main.py --all")
    print("="*60)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce thesis results: Explainable Deep Learning "
            "for Breast Cancer Diagnosis (Um-e-Farwa, COMSATS 2023)"
        )
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete pipeline (requires MIAS dataset)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo mode — survey figures only, no dataset needed"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Evaluate only (requires trained models)"
    )
    parser.add_argument(
        "--survey_only", action="store_true",
        help="Survey analysis figures only"
    )
    parser.add_argument(
        "--gradcam_only", action="store_true",
        help="Grad-CAM only (requires trained VGG-16)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help=f"Path to MIAS dataset (default: {DATA_DIR})"
    )
    args = parser.parse_args()
    
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    start = time.time()
    
    print("\n" + "="*60)
    print("EXPLAINABLE DEEP LEARNING FOR BREAST CANCER DIAGNOSIS")
    print("A Comparative Study with Expert Radiologist Interpretations")
    print("Author: Um-e-Farwa | COMSATS University Islamabad | 2023")
    print("Preprint: ssrn.com/abstract=6583028")
    print("="*60)
    
    if args.demo or not any([
        args.all, args.eval_only, args.survey_only, args.gradcam_only
    ]):
        run_demo()
        return
    
    if args.survey_only:
        step5_survey()
        return
    
    # Steps requiring dataset
    X_train, X_test, y_train, y_test = step1_load_data()
    
    if args.all:
        step2_train_all(X_train, X_test, y_train, y_test)
        step3_evaluate_all(X_test, y_test)
        step4_gradcam(X_test, y_test)
        step5_survey()
    
    elif args.eval_only:
        step3_evaluate_all(X_test, y_test)
    
    elif args.gradcam_only:
        step4_gradcam(X_test, y_test)
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"All figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
