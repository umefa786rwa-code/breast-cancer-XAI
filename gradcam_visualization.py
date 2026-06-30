"""
gradcam_visualization.py
========================
Grad-CAM heat map generation for VGG-16 predictions.

This is the CORE CONTRIBUTION of the thesis (Section 3.9, Figure 3.8).

Grad-CAM (Gradient-weighted Class Activation Mapping) uses gradients
from the final convolutional layer to generate spatially resolved heat
maps showing WHICH regions of the mammogram drove each prediction.

These heat maps were shown to expert radiologists in the clinical
validation survey. The central finding:
    → Only 30% of radiologists could correctly identify the disease
    → 70% could NOT use the AI output despite 98% model accuracy

This quantifies the clinical trust gap in medical AI.

Reference: Selvaraju et al. (2017). Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization. ICCV 2017.

Usage:
    python gradcam_visualization.py --image path/to/mammogram.pgm
    python gradcam_visualization.py --batch data/mias/MALIGNANT/

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from preprocessing import load_pgm_image, preprocess_image, IMAGE_SIZE


# ── Grad-CAM Core ─────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.
    
    Algorithm (Selvaraju et al. 2017, thesis Section 1.2.3):
    
    1. Forward pass: get model prediction for input image
    2. Select target class (predicted or specified)
    3. Compute gradient of class score with respect to 
       final convolutional layer feature maps
    4. Global average pool the gradients → importance weights (α)
    5. Weighted combination of feature maps → raw heat map
    6. Apply ReLU (only positive influences)
    7. Resize and normalise to [0, 1]
    8. Overlay as colour heat map on original image
    
    Key distinction from CAM (thesis Section 1.2.3):
        Grad-CAM uses global average pooling of GRADIENTS
        rather than weights, making it applicable to ANY CNN
        architecture without modification.
    """
    
    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        """
        Initialise Grad-CAM for a trained model.
        
        Args:
            model:      Trained Keras model (VGG-16, ResNet-50, or CNN)
            layer_name: Name of final convolutional layer.
                        If None, auto-detects the last Conv2D layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        # Build gradient model — outputs both the conv layer
        # activations AND the final class predictions
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(self.layer_name).output,
                model.output
            ]
        )
        print(f"Grad-CAM initialised. Target layer: {self.layer_name}")
    
    def _find_last_conv_layer(self) -> str:
        """
        Auto-detect the name of the last Conv2D layer.
        
        The final convolutional layer contains the most task-specific
        spatial information — ideal for class activation mapping
        (thesis Section 1.2.3).
        
        Returns:
            Name of last Conv2D layer
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found in model.")
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: int = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM heat map for a single image.
        
        Args:
            image:     Preprocessed image array (224, 224, 3), 
                       normalised to [0, 1], shape must include batch dim
                       OR be (224, 224, 3) — batch dim added automatically
            class_idx: Target class index (None = use predicted class)
            eps:       Small value to prevent division by zero
            
        Returns:
            Heat map array of shape (224, 224), values in [0, 1]
            Higher values = regions MORE important for the prediction
        """
        # Add batch dimension if needed
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        image_tensor = tf.cast(image, tf.float32)
        
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            
            # Forward pass — get conv features and predictions
            conv_outputs, predictions = self.grad_model(image_tensor)
            
            # Select target class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Score for target class
            class_score = predictions[:, class_idx]
        
        # Compute gradients of class score w.r.t. conv feature maps
        # Shape: (1, H, W, C) where H,W = spatial dims, C = channels
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global average pool gradients over spatial dimensions
        # → importance weights α for each feature map channel
        # Shape: (1, 1, C)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by their importance
        conv_outputs = conv_outputs[0]     # Remove batch dim → (H, W, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)      # → (H, W)
        
        # Apply ReLU — only keep positive contributions
        # (negative contributions decrease target class score)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalise to [0, 1]
        heatmap = heatmap.numpy()
        heatmap = (heatmap - heatmap.min()) / (
            heatmap.max() - heatmap.min() + eps
        )
        
        return heatmap
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> tuple:
        """
        Overlay Grad-CAM heat map on the original mammogram.
        
        Reproduces the visualisation in thesis Figure 3.8.
        Red regions = HIGH importance (model focuses here)
        Blue regions = LOW importance
        
        Args:
            heatmap:        Grad-CAM output (H, W), values in [0, 1]
            original_image: Original image array (H, W, 3) or (H, W)
            alpha:          Blending factor (0=original, 1=heatmap only)
            colormap:       OpenCV colormap (JET = blue→red)
            
        Returns:
            heatmap_colored: Coloured heat map (H, W, 3), uint8
            superimposed:    Blended overlay (H, W, 3), uint8
        """
        # Get target size from original image
        h = original_image.shape[0]
        w = original_image.shape[1]
        
        # Resize heat map to match original image size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colour map (JET: blue=cold=unimportant, red=hot=important)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Prepare original image as uint8 RGB
        if original_image.dtype != np.uint8:
            orig_uint8 = np.uint8(255 * original_image)
        else:
            orig_uint8 = original_image
        
        if orig_uint8.ndim == 2:
            orig_uint8 = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
        
        # Blend heat map with original
        superimposed = cv2.addWeighted(
            orig_uint8, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        
        return heatmap_colored, superimposed


# ── Visualisation helpers ──────────────────────────────────────────────────────
def visualise_gradcam(
    model: tf.keras.Model,
    image_path: str,
    class_names: list = ["Benign", "Malignant"],
    save_path: str = None,
    layer_name: str = None
):
    """
    Full Grad-CAM pipeline for a single mammogram image.
    
    Shows 4 panels (reproducing thesis Figure 3.8 layout):
        1. Original mammogram
        2. Grad-CAM heat map
        3. Superimposed overlay
        4. Contour overlay (ROI boundaries)
    
    Args:
        model:       Trained Keras model
        image_path:  Path to mammogram file (.pgm, .png, .jpg)
        class_names: List of class label strings
        save_path:   Optional path to save the figure
        layer_name:  Target conv layer (None = auto-detect)
    """
    # Load and preprocess image
    raw = load_pgm_image(image_path)
    processed = preprocess_image(raw)
    
    # Get prediction
    input_tensor = np.expand_dims(processed, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class] * 100
    
    # Compute Grad-CAM
    gcam = GradCAM(model, layer_name)
    heatmap = gcam.compute_heatmap(processed, class_idx=pred_class)
    heatmap_colored, superimposed = gcam.overlay_heatmap(heatmap, raw)
    
    # Create contour overlay (4th panel — ROI boundaries)
    heatmap_resized = cv2.resize(heatmap, (raw.shape[1], raw.shape[0]))
    _, contour_mask = cv2.threshold(
        np.uint8(255 * heatmap_resized), 127, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    contour_img = raw_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 100), 2)
    
    # Plot 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    panels = [
        (raw, "gray", "Original Mammogram"),
        (heatmap_colored, None, "Grad-CAM Heat Map"),
        (superimposed, None, "Superimposed Overlay"),
        (contour_img, None, "ROI Contours"),
    ]
    
    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    
    # Add prediction annotation
    label = class_names[pred_class]
    color = "#C0392B" if pred_class == 1 else "#2A7A2A"
    fig.suptitle(
        f"Prediction: {label} ({confidence:.1f}% confidence) | "
        f"File: {Path(image_path).name}",
        fontsize=12, color=color, fontweight="bold"
    )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return heatmap, pred_class, confidence


def visualise_batch(
    model: tf.keras.Model,
    image_dir: str,
    n_images: int = 8,
    save_path: str = "figures/gradcam_batch.png",
    layer_name: str = None
):
    """
    Generate Grad-CAM visualisations for a batch of images.
    
    Reproduces the grid layout of thesis Figure 3.8
    (4 rows × 4 columns showing original + heat map + 
    overlay + contour for multiple mammograms).
    
    Args:
        model:     Trained Keras model
        image_dir: Directory containing mammogram images
        n_images:  Number of images to visualise
        save_path: Path to save the grid figure
        layer_name: Target conv layer (None = auto-detect)
    """
    # Get image files
    extensions = (".pgm", ".png", ".jpg", ".jpeg")
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(extensions)
    ][:n_images]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    gcam = GradCAM(model, layer_name)
    
    fig, axes = plt.subplots(
        len(image_files), 4,
        figsize=(18, 4.5 * len(image_files))
    )
    
    if len(image_files) == 1:
        axes = [axes]
    
    class_names = ["Benign", "Malignant"]
    colors = ["#2A7A2A", "#C0392B"]
    
    for i, fpath in enumerate(image_files):
        try:
            raw = load_pgm_image(fpath)
            processed = preprocess_image(raw)
            
            pred = model.predict(
                np.expand_dims(processed, 0), verbose=0
            )[0]
            pred_class = np.argmax(pred)
            confidence = pred[pred_class] * 100
            
            heatmap = gcam.compute_heatmap(processed, pred_class)
            heatmap_colored, superimposed = gcam.overlay_heatmap(
                heatmap, raw
            )
            
            # Contour panel
            hm_resized = cv2.resize(
                heatmap, (raw.shape[1], raw.shape[0])
            )
            _, mask = cv2.threshold(
                np.uint8(255 * hm_resized), 127, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            contour_img = raw_rgb.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 100), 2)
            
            panels = [
                (raw, "gray"),
                (heatmap_colored, None),
                (superimposed, None),
                (contour_img, None),
            ]
            
            for j, (img, cmap) in enumerate(panels):
                axes[i][j].imshow(img, cmap=cmap)
                axes[i][j].axis("off")
                if j == 0:
                    label = class_names[pred_class]
                    col = colors[pred_class]
                    axes[i][j].set_title(
                        f"{Path(fpath).name}\n{label} "
                        f"({confidence:.0f}%)",
                        fontsize=9, color=col
                    )
        
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
    
    # Column headers
    col_titles = [
        "Original Mammogram", "Grad-CAM Heat Map",
        "Superimposed Overlay", "ROI Contours"
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(
            title, fontsize=11, fontweight="bold",
            pad=10, color="#2A2460"
        )
    
    plt.suptitle(
        "Grad-CAM Visualisations — VGG-16 on MIAS Mammography Dataset\n"
        "Red regions indicate areas most important for the model's decision",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nBatch visualisation saved to: {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heat maps for breast cancer diagnosis"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to single mammogram image"
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Directory of mammogram images for batch processing"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="saved_models/vgg16_best.h5",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        help="Target conv layer name (default: auto-detect last Conv2D)"
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="Number of images for batch mode (default: 8)"
    )
    args = parser.parse_args()
    
    # Load trained model
    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print("Model loaded successfully.")
    
    if args.image:
        print(f"\nGenerating Grad-CAM for: {args.image}")
        visualise_gradcam(
            model, args.image,
            save_path=f"figures/gradcam_{Path(args.image).stem}.png",
            layer_name=args.layer
        )
    
    elif args.batch:
        print(f"\nGenerating batch Grad-CAM from: {args.batch}")
        visualise_batch(
            model, args.batch,
            n_images=args.n,
            layer_name=args.layer
        )
    
    else:
        print("\nUsage examples:")
        print("  Single image:  python gradcam_visualization.py --image data/mias/MALIGNANT/mdb001.pgm")
        print("  Batch:         python gradcam_visualization.py --batch data/mias/MALIGNANT/ --n 8")
        print("\nThe heat maps show which mammogram regions drove the model's")
        print("prediction. Red = high importance, Blue = low importance.")
        print("\nThesis finding: 70% of expert radiologists could NOT correctly")
        print("identify disease from these Grad-CAM outputs despite 98% accuracy.")


if __name__ == "__main__":
    main()
