"""
models.py
=========
Three CNN architectures used in thesis.

Matches thesis methodology (Sections 1.2.1, 1.2.2, 3.8):
1. Baseline CNN         — 96% accuracy
2. ResNet-50            — 97% accuracy  
3. VGG-16 (best model) — 98% accuracy, 99% precision, 97% MCC

All models use transfer learning with ImageNet weights,
with upper layers fine-tuned on MIAS mammography data.

Author: Um-e-Farwa | COMSATS University Islamabad | 2023
Preprint: https://ssrn.com/abstract=6583028
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam


# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224, 3)   # Input shape for all models
NUM_CLASSES   = 2               # Binary: benign vs malignant
LEARNING_RATE = 1e-4            # Adam optimiser learning rate
DROPOUT_RATE  = 0.5             # Dropout for regularisation


# ── Model 1: Baseline CNN ──────────────────────────────────────────────────────
def build_baseline_cnn(
    input_shape: tuple = IMAGE_SIZE,
    num_classes: int   = NUM_CLASSES
) -> tf.keras.Model:
    """
    Baseline CNN architecture (thesis Table 3.4).
    
    Results from thesis:
        Accuracy:    96%
        Precision:   95%
        Sensitivity: 97%
        Specificity: 95%
        MCC:         91%
        Training time: 12 minutes
    
    Architecture:
        Input → Conv2D → MaxPool → Conv2D → MaxPool → 
        Flatten → Dense(256) → Dropout → Dense(128) → 
        Dropout → Output(Softmax)
    
    Args:
        input_shape: Image dimensions (224, 224, 3)
        num_classes: Number of output classes (2 for binary)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", 
                      padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Fully connected
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(128, activation="relu"),
        layers.Dropout(DROPOUT_RATE),

        # Output
        layers.Dense(num_classes, activation="softmax")
    ], name="baseline_cnn")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Model 2: ResNet-50 ────────────────────────────────────────────────────────
def build_resnet50(
    input_shape: tuple = IMAGE_SIZE,
    num_classes: int   = NUM_CLASSES,
    freeze_layers: int = 140
) -> tf.keras.Model:
    """
    ResNet-50 with transfer learning (thesis Table 3.5).
    
    Won the 2015 ILSVRC competition. Uses residual connections
    with batch normalisation to enable very deep networks
    (Section 1.2.2).
    
    Results from thesis:
        Accuracy:    97%
        Precision:   98%
        Sensitivity: 97%
        Specificity: 98%
        MCC:         95%
        Training time: 25 minutes
    
    Transfer learning strategy (thesis Section 3.8):
        - Lower layers: FROZEN (general ImageNet features)
        - Upper layers: FINE-TUNED (mammography-specific features)
    
    Args:
        input_shape:   Image dimensions (224, 224, 3)
        num_classes:   Number of output classes
        freeze_layers: Number of base layers to freeze
        
    Returns:
        Compiled Keras model
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze lower layers (general features from ImageNet)
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    # Fine-tune upper layers (mammography-specific features)
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True

    # Add classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="resnet50_mammography"
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Model 3: VGG-16 (Best Model) ─────────────────────────────────────────────
def build_vgg16(
    input_shape: tuple = IMAGE_SIZE,
    num_classes: int   = NUM_CLASSES,
    freeze_layers: int = 15
) -> tf.keras.Model:
    """
    VGG-16 with transfer learning — BEST MODEL (thesis Table 3.6).
    
    Finished second in 2014 ILSVRC. Uses uniform 3x3 convolutions
    throughout, making it excellent for texture-based feature learning
    in mammography (Section 1.2.1).
    
    This model was selected for Grad-CAM visualisation in the thesis.
    
    Results from thesis:
        Accuracy:      98%  ← Best
        Precision:     99%  ← Best
        Sensitivity:   98%
        Specificity:   99%  ← Best
        MCC:           97%  ← Best
        Training time: 10.25 minutes  ← Fastest
    
    Transfer learning strategy (thesis Section 3.8):
        - Lower layers: FROZEN (general ImageNet features)
        - Upper layers: FINE-TUNED (mammography-specific features)
    
    Args:
        input_shape:   Image dimensions (224, 224, 3)
        num_classes:   Number of output classes
        freeze_layers: Number of base layers to freeze
        
    Returns:
        Compiled Keras model
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze lower layers (general features)
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    # Fine-tune upper layers (mammography-specific)
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True

    # Add classification head
    # Replaces VGG-16's original fully-connected layers
    # with mammography-optimised dense layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="vgg16_mammography"
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_model(name: str) -> tf.keras.Model:
    """
    Factory function to get any model by name.
    
    Args:
        name: One of "cnn", "resnet50", "vgg16"
        
    Returns:
        Compiled Keras model
    """
    builders = {
        "cnn":     build_baseline_cnn,
        "resnet50": build_resnet50,
        "vgg16":   build_vgg16
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Choose from {list(builders)}")
    return builders[name]()


def print_model_summary(name: str = "vgg16"):
    """Print architecture summary for any model."""
    model = get_model(name)
    print(f"\n{'='*60}")
    print(f"Model: {name.upper()}")
    print('='*60)
    model.summary()
    trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    total = sum(
        tf.size(w).numpy() for w in model.weights
    )
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")


if __name__ == "__main__":
    print("Models module loaded.")
    print("\nThesis results summary:")
    print("-" * 55)
    print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'MCC':>8}")
    print("-" * 55)
    print(f"{'CNN':<12} {'96%':>10} {'95%':>10} {'91%':>8}")
    print(f"{'ResNet-50':<12} {'97%':>10} {'98%':>10} {'95%':>8}")
    print(f"{'VGG-16':<12} {'98%':>10} {'99%':>10} {'97%':>8} ← BEST")
    print("-" * 55)
    print("\nTo build VGG-16: from models import build_vgg16")
