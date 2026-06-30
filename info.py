#!/usr/bin/env python3
"""
Explainable Deep Learning for Breast Cancer Diagnosis Using Grad-CAM
A Comparative Study with Expert Radiologist Interpretations

Author: Um-e-Farwa
Institution: COMSATS University Islamabad, Pakistan
MS in Health Informatics (CGPA: 3.75)
Year: 2023

Preprint: https://ssrn.com/abstract=6583028
Manuscript: Under review at Informatics in Medicine Unlocked (Elsevier)

CENTRAL FINDING:
VGG-16 achieved 98% accuracy on MIAS mammography dataset.
Despite this, 70% of expert radiologists could NOT correctly identify
the disease from Grad-CAM outputs — quantifying the clinical trust gap.

USAGE:
    python train.py                          # Train all three models
    python gradcam_visualization.py          # Generate Grad-CAM heat maps
    python evaluate.py                       # Evaluate and plot results
    python requirements.txt                  # Install dependencies

DATASET:
    Download MIAS dataset from Kaggle:
    https://www.kaggle.com/datasets/kmader/mias-mammography
    Place in: data/mias/
"""

# ============================================================
# File: requirements.txt (install with: pip install -r requirements.txt)
# ============================================================
REQUIREMENTS = """
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
Pillow>=8.3.0
scipy>=1.7.0
seaborn>=0.11.0
"""

print("See requirements.txt for dependencies.")
print("Install with: pip install -r requirements.txt")
