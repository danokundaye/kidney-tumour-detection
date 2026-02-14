# Hybrid YOLO-UNet Architecture for Kidney Tumour Detection, Segmentation, and Classification

**Author:** Daniel Osaseri Okundaye  
**Institution:** MIVA Open University, Abuja, Nigeria  
**Supervisor:** Isiekwene Chinyere Chioma  
**Degree:** B.Sc. (Hons) Computer Science  
**Year:** 2025  


## Project Overview

This project implements a three-stage deep learning pipeline for automated detection, segmentation, and classification of kidney tumours from CT scans. The system is designed to support radiologists in resource-constrained healthcare settings by reducing manual workload and improving diagnostic consistency.

### Pipeline Summary

| Stage | Model | Task | Target Metric |
|-------|-------|------|---------------|
| 1 | YOLOv8 | Kidney region detection | Sensitivity >90% |
| 2 | 3D U-Net + ResNet50 | Tumour segmentation | Dice ≥0.80 |
| 3 | EfficientNet-B0 | Benign/malignant classification | Accuracy >85% |
| Post | SHAP | Explainability | Visual attribution maps |

### Dataset
- **KiTS21** (Kidney Tumour Segmentation Challenge 2021)
- 300 anonymized pre-operative CT cases
- NIfTI format, 512×512 pixels per slice
- Patient-level split: 110 (detection) / 120 (segmentation) / 
  70 (testing)

---

## Repository Structure
```
kidney-tumour-detection/
│
├── data/                    # Local data samples only (2-3 cases)
│   ├── raw/                 # Original NIfTI files
│   └── processed/           # Preprocessed outputs
│
├── notebooks/               # Google Colab training notebooks
│
├── src/                     # Source code
│   ├── preprocessing/       # NIfTI loading, slicing, augmentation
│   ├── detection/           # YOLOv8 training and inference
│   ├── segmentation/        # U-Net training and inference
│   ├── classification/      # EfficientNet training and inference
│   ├── explainability/      # SHAP integration
│   └── evaluation/          # Metrics and reporting
│
├── configs/                 # Training configuration files
├── outputs/                 # Predictions and visualizations
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs and CSV metrics
├── tests/                   # Module verification scripts
├── requirements-local.txt   # Local development dependencies
└── requirements-colab.txt   # Colab training dependencies (Phase 2)
```

---

## Local Development Setup

These steps are intended to set up local machine for preprocessing and code editing only. Model training happens on Google Colab.

### Prerequisites
- Windows 10/11
- Python 3.11.x
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/danokundaye/kidney-tumour-detection.git
cd kidney-tumour-detection
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate    # GitBash on Windows
```

**3. Install local dependencies**
```bash
pip install -r requirements-local.txt
```

---

## Implementation Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Local VS Code Setup | ✅ Complete |
| 2 | Google Colab + Drive Setup | ✅ Complete |
| 3 | Dataset Download and Verification | ✅ Complete |
| 4 | Preprocessing Pipeline | ⏳ Pending |
| 5 | YOLOv8 Detection Training | ⏳ Pending |
| 6 | U-Net Segmentation Training | ⏳ Pending |
| 7 | EfficientNet Classification Training | ⏳ Pending |
| 8 | SHAP Integration | ⏳ Pending |
| 9 | End-to-End Pipeline Integration | ⏳ Pending |
| 10 | Evaluation and Metrics | ⏳ Pending |

---

## Academic Context

This system is a **proof of concept** and is not intended for clinical deployment. All experiments use the publicly available, pre-anonymized KiTS21 dataset. Results will be reported in Chapter 4 of the project.

This work contributes to:
- **SDG 3** — Good Health and Well-being
- **SDG 9** — Industry, Innovation and Infrastructure