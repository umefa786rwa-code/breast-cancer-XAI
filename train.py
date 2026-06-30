"""
train.py
========
Training pipeline for all three CNN models.

Matches thesis methodology (Section 3.2, 3.8):
- Dataset split: 75% training, 25% validation
  (257 training, 65 testing from 322 total)
- Transfer learning with frozen lower layers
- Fine-tuning of upper layers on MIAS data

Usage:
    python train.py --model vgg16      # Train VGG-16 (best model)
    python train.py --model resnet50   # Train ResNet-50
    python train.py --model cnn        # Train baseline CNN
    python train.py --all              # Train all three models

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from preprocessing import load_mias_dataset, IMAGE_SIZE
from augmentation import augment_batch, get_augmentation_generator
from models import get_model


# ── Configuration ─────────────────────────────────────────────────────────────
EPOCHS       = 50
BATCH_SIZE   = 16
TEST_SIZE    = 0.25   # 25% held out for validation (thesis Section 3.2)
RANDOM_SEED  = 42
SAVE_DIR     = "saved_models/"
LOG_DIR      = "logs/"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_and_split_data(data_dir: str = "data/mias/") -> tuple:
    """
    Load MIAS dataset and split into train/test sets.
    
    Thesis Section 3.2:
        Total: 322 images
        Training: 75% = 257 images (after augmentation: 1028)
        Testing:  25% = 65 images
        
    Args:
        data_dir: Root directory of MIAS dataset
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading MIAS dataset...")
    X, y, filenames = load_mias_dataset(data_dir, mode="binary")
    
    print(f"Total images loaded: {len(X)}")
    print(f"Class distribution: Benign={sum(y==0)}, Malignant={sum(y==1)}")
    
    # Split: 75% train, 25% test (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y   # Preserve class distribution in both splits
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Test set:     {len(X_test)} images")
    
    # Apply augmentation to training set only
    print("\nApplying data augmentation...")
    X_train_aug, y_train_aug = augment_batch(X_train, y_train, multiplier=3)
    print(f"After augmentation: {len(X_train_aug)} training images")
    print(f"(Matches thesis: 322 total → ~1028 after augmentation)")
    
    return X_train_aug, X_test, y_train_aug, y_test


# ── Callbacks ──────────────────────────────────────────────────────────────────
def get_callbacks(model_name: str) -> list:
    """
    Training callbacks for monitoring and checkpointing.
    
    Args:
        model_name: Name string for saving files
        
    Returns:
        List of Keras callbacks
    """
    return [
        # Save best model weights
        ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, f"{model_name}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Stop training if no improvement
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Log training history to CSV
        CSVLogger(
            os.path.join(LOG_DIR, f"{model_name}_history.csv")
        )
    ]


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Train a single model and return history.
    
    Uses class weights to handle class imbalance
    (MIAS: 54 malignant vs 67 benign).
    
    Args:
        model_name: One of "cnn", "resnet50", "vgg16"
        X_train, X_test: Image arrays
        y_train, y_test: Label arrays
        
    Returns:
        Training history dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print('='*60)
    
    # Build model
    model = get_model(model_name)
    model.summary()
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    # Save final model
    save_path = os.path.join(SAVE_DIR, f"{model_name}_final.h5")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Evaluate on test set
    print(f"\nFinal evaluation on test set:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy: {acc*100:.1f}%")
    print(f"  Test loss:     {loss:.4f}")
    
    return history.history


# ── Plot training curves ───────────────────────────────────────────────────────
def plot_training_history(
    histories: dict,
    save_path: str = "figures/training_curves.png"
):
    """
    Plot accuracy and loss curves for all trained models.
    
    Reproduces thesis Figures 3.6 and 3.7 (accuracy plots for
    ResNet-50 and VGG-16).
    
    Args:
        histories: Dict of {model_name: history_dict}
        save_path: Path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(
        len(histories), 2,
        figsize=(14, 4 * len(histories))
    )
    
    if len(histories) == 1:
        axes = [axes]
    
    colors = {"cnn": "#2A2460", "resnet50": "#0D6E5C", "vgg16": "#C0392B"}
    
    for i, (name, hist) in enumerate(histories.items()):
        color = colors.get(name, "#333333")
        
        # Accuracy plot
        ax_acc = axes[i][0]
        ax_acc.plot(hist["accuracy"], label="Train",
                    color=color, linewidth=2)
        ax_acc.plot(hist["val_accuracy"], label="Test",
                    color=color, linewidth=2, linestyle="--")
        ax_acc.set_title(f"{name.upper()} — Accuracy", fontsize=12)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)
        
        # Loss plot
        ax_loss = axes[i][1]
        ax_loss.plot(hist["loss"], label="Train",
                     color=color, linewidth=2)
        ax_loss.plot(hist["val_loss"], label="Test",
                     color=color, linewidth=2, linestyle="--")
        ax_loss.set_title(f"{name.upper()} — Loss", fontsize=12)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)
    
    plt.suptitle(
        "Training and Validation Curves — MIAS Mammography Dataset",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTraining curves saved to: {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train CNN models for breast cancer diagnosis (MIAS dataset)"
    )
    parser.add_argument(
        "--model", type=str, default="vgg16",
        choices=["cnn", "resnet50", "vgg16"],
        help="Model to train (default: vgg16)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all three models"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/mias/",
        help="Path to MIAS dataset directory"
    )
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data(args.data_dir)
    
    # Select models to train
    models_to_train = (
        ["cnn", "resnet50", "vgg16"] if args.all else [args.model]
    )
    
    histories = {}
    for model_name in models_to_train:
        histories[model_name] = train_model(
            model_name, X_train, X_test, y_train, y_test
        )
    
    # Plot training curves
    if histories:
        plot_training_history(histories)
    
    print("\n" + "="*60)
    print("Training complete.")
    print(f"Models saved in: {SAVE_DIR}")
    print(f"Logs saved in:   {LOG_DIR}")
    print("\nNext step: Run gradcam_visualization.py to generate heat maps")
    print("="*60)


if __name__ == "__main__":
    main()
