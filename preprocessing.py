"""
preprocessing.py
================
Preprocessing pipeline for MIAS mammography dataset.

Matches thesis methodology (Section 3.4):
1. Noise reduction
2. Contrast enhancement
3. Label suppression
4. Pectoral muscle removal

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)          # Standard input size for VGG-16 and ResNet-50
MIAS_PATH  = "data/mias/"        # Path to MIAS dataset PGM files
OUTPUT_PATH = "data/processed/"  # Path to save processed images

# MIAS class mapping (from thesis Table 3.2)
CLASS_MAP = {
    "NORM": 0,   # Normal - 189 images
    "BENIGN": 1, # Benign abnormality - 67 images
    "MALIGNANT": 2  # Malignant abnormality - 54 images
}

# For binary classification (thesis uses benign vs malignant)
BINARY_CLASS_MAP = {
    "BENIGN": 0,
    "MALIGNANT": 1
}


def load_pgm_image(filepath: str) -> np.ndarray:
    """
    Load a PGM (Portable Gray Map) image from MIAS dataset.
    
    MIAS images are 1024x1024 pixels at 200-micron resolution
    (digitised from 50-micron, then reduced as per thesis Section 3.1).
    
    Args:
        filepath: Path to .pgm file
        
    Returns:
        numpy array of shape (1024, 1024) with pixel values 0-255
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    return img


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """
    Step 1: Noise reduction using Gaussian blur.
    
    Removes high-frequency noise from mammogram while
    preserving clinically relevant features and edges.
    
    Args:
        image: Grayscale mammogram array
        
    Returns:
        Noise-reduced image
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Step 2: Contrast enhancement using CLAHE.
    
    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    enhances local contrast in mammograms, making masses
    and calcifications more visible for model training.
    
    Args:
        image: Grayscale mammogram array
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def suppress_labels(image: np.ndarray) -> np.ndarray:
    """
    Step 3: Label suppression.
    
    Removes radiological labels and annotations from mammogram borders
    that could confuse the model during training.
    
    Args:
        image: Grayscale mammogram array
        
    Returns:
        Image with border labels removed
    """
    # Create mask - set border region to black (labels are typically in corners)
    h, w = image.shape
    border = 30  # pixels to remove from each border
    cleaned = image.copy()
    cleaned[:border, :] = 0      # top border
    cleaned[-border:, :] = 0     # bottom border
    cleaned[:, :border] = 0      # left border
    cleaned[:, -border:] = 0     # right border
    return cleaned


def remove_pectoral_muscle(image: np.ndarray) -> np.ndarray:
    """
    Step 4: Pectoral muscle removal.
    
    The pectoral muscle appears with similar intensity to breast
    abnormalities in MLO-view mammograms. Removing it reduces
    false positives in classification (thesis Section 3.4).
    
    Uses threshold + contour detection to identify and mask the
    pectoral region (typically upper-left or upper-right corner).
    
    Args:
        image: Preprocessed grayscale mammogram
        
    Returns:
        Image with pectoral muscle region suppressed
    """
    # Threshold to find bright regions
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return image
    
    # The pectoral muscle is usually the largest bright region
    # in the corner — mask it out
    result = image.copy()
    h, w = image.shape
    
    # Simple approach: if large bright region is in corner, suppress it
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0.05 * h * w:  # More than 5% of image area
            x, y, cw, ch = cv2.boundingRect(contour)
            # Only suppress if it starts near a corner (pectoral characteristic)
            if x < w * 0.15 or x + cw > w * 0.85:
                if y < h * 0.25:  # Upper portion of image
                    cv2.drawContours(result, [contour], -1, 0, -1)
    
    return result


def preprocess_image(
    image: np.ndarray,
    target_size: tuple = IMAGE_SIZE
) -> np.ndarray:
    """
    Full preprocessing pipeline.
    
    Applies all 4 preprocessing steps from thesis Section 3.4,
    then resizes to model input size and converts to 3-channel
    (required for VGG-16 and ResNet-50 pretrained on ImageNet).
    
    Args:
        image: Raw grayscale mammogram from MIAS
        target_size: Output size (224, 224) for transfer learning models
        
    Returns:
        Preprocessed RGB image array of shape (224, 224, 3), 
        normalised to [0, 1]
    """
    # Apply 4-step preprocessing pipeline
    img = reduce_noise(image)
    img = enhance_contrast(img)
    img = suppress_labels(img)
    img = remove_pectoral_muscle(img)
    
    # Resize to model input size
    img = cv2.resize(img, target_size)
    
    # Convert grayscale to 3-channel RGB
    # (VGG-16 and ResNet-50 expect 3 channels)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Normalise to [0, 1]
    img_normalised = img_rgb.astype(np.float32) / 255.0
    
    return img_normalised


def visualise_preprocessing(image_path: str, save_path: str = None):
    """
    Visualise each step of the preprocessing pipeline.
    
    Useful for debugging and for the thesis figures.
    
    Args:
        image_path: Path to raw MIAS PGM file
        save_path: Optional path to save visualisation
    """
    raw = load_pgm_image(image_path)
    
    steps = [
        ("Original", raw),
        ("After Noise Reduction", reduce_noise(raw)),
        ("After Contrast Enhancement", enhance_contrast(reduce_noise(raw))),
        ("After Label Suppression",
         suppress_labels(enhance_contrast(reduce_noise(raw)))),
        ("After Pectoral Removal",
         remove_pectoral_muscle(
             suppress_labels(enhance_contrast(reduce_noise(raw))))),
    ]
    
    fig, axes = plt.subplots(1, len(steps), figsize=(20, 4))
    for ax, (title, img) in zip(axes, steps):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    
    plt.suptitle("MIAS Mammogram Preprocessing Pipeline", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def load_mias_dataset(
    data_dir: str = MIAS_PATH,
    mode: str = "binary"
) -> tuple:
    """
    Load and preprocess the full MIAS dataset.
    
    Expects MIAS images organised as:
        data/mias/
            MALIGNANT/  (54 images)
            BENIGN/     (67 images)
            NORMAL/     (201 images - used for binary: excluded)
    
    For binary classification (thesis approach):
        class 0 = BENIGN
        class 1 = MALIGNANT
    
    Args:
        data_dir: Root directory containing MIAS images
        mode: "binary" (benign vs malignant) or "multiclass" (3 classes)
        
    Returns:
        X: numpy array of shape (n_samples, 224, 224, 3)
        y: numpy array of class labels
        filenames: list of image filenames
    """
    X, y, filenames = [], [], []
    
    if mode == "binary":
        classes = {"BENIGN": 0, "MALIGNANT": 1}
    else:
        classes = {"NORMAL": 0, "BENIGN": 1, "MALIGNANT": 2}
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found — skipping")
            continue
            
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".pgm", ".png", ".jpg")):
                fpath = os.path.join(class_dir, fname)
                try:
                    raw = load_pgm_image(fpath)
                    processed = preprocess_image(raw)
                    X.append(processed)
                    y.append(label)
                    filenames.append(fname)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
    
    return np.array(X), np.array(y), filenames


if __name__ == "__main__":
    print("Preprocessing module loaded.")
    print(f"Target image size: {IMAGE_SIZE}")
    print(f"MIAS dataset path: {MIAS_PATH}")
    print("\nTo test preprocessing on a single image:")
    print("  from preprocessing import visualise_preprocessing")
    print("  visualise_preprocessing('data/mias/MALIGNANT/mdb001.pgm')")
