"""
augmentation.py
===============
Data augmentation for MIAS mammography dataset.

Matches thesis methodology (Section 3.7):
- Left to right flip
- Up to down flip  
- Rotation at 90 degrees
- Result: 322 images → 1028 images (3x augmentation)

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_augmentation_generator() -> ImageDataGenerator:
    """
    Create Keras ImageDataGenerator matching thesis augmentation.
    
    Thesis Table 3.3:
        Original MIAS: 322 images
        After augmentation: 1028 images
        Operations: horizontal flip, vertical flip, 90-degree rotation
        
    Returns:
        Configured ImageDataGenerator for training
    """
    return ImageDataGenerator(
        horizontal_flip=True,       # Left to right flip (thesis)
        vertical_flip=True,         # Up to down flip (thesis)
        rotation_range=90,          # 90-degree rotation (thesis)
        rescale=1.0 / 255.0,        # Normalise to [0,1]
        fill_mode="nearest"
    )


def get_validation_generator() -> ImageDataGenerator:
    """
    Validation generator — no augmentation, only normalisation.
    
    Returns:
        ImageDataGenerator for validation (rescale only)
    """
    return ImageDataGenerator(rescale=1.0 / 255.0)


def augment_batch(X: np.ndarray, y: np.ndarray, 
                  multiplier: int = 3) -> tuple:
    """
    Manually augment a dataset batch to match thesis ratio.
    
    Thesis used 322 → 1028 images (approximately 3x).
    
    Args:
        X: Input images array (n, 224, 224, 3) — already normalised
        y: Labels array (n,)
        multiplier: How many augmented copies to add per original
        
    Returns:
        X_aug: Augmented images (n * (1 + multiplier), 224, 224, 3)
        y_aug: Augmented labels
    """
    augmented_images = [X]
    augmented_labels = [y]
    
    for _ in range(multiplier):
        batch = []
        for img in X:
            # Apply random augmentation from thesis set
            choice = np.random.randint(3)
            if choice == 0:
                # Horizontal flip (left to right)
                batch.append(np.fliplr(img))
            elif choice == 1:
                # Vertical flip (up to down)
                batch.append(np.flipud(img))
            else:
                # 90-degree rotation
                batch.append(np.rot90(img, k=1))
        
        augmented_images.append(np.array(batch))
        augmented_labels.append(y)
    
    X_aug = np.concatenate(augmented_images, axis=0)
    y_aug = np.concatenate(augmented_labels, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X_aug))
    return X_aug[indices], y_aug[indices]


if __name__ == "__main__":
    print("Augmentation module loaded.")
    print("Thesis augmentation: horizontal flip + vertical flip + 90° rotation")
    print("Result: 322 → 1028 images (3x augmentation)")
