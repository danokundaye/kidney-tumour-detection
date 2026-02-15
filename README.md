# Hybrid YOLO-UNet Architecture for Kidney Tumour Detection, Segmentation, and Classification

**Author:** Daniel Osaseri Okundaye  
**Institution:** Miva Open University, Abuja, Nigeria  
**Supervisor:** Isiekwene Chinyere Chioma  
**Degree:** B.Sc. (Hons) Computer Science  
**Year:** 2026


## Project Overview

This project implements a three-stage deep learning pipeline for automated detection, segmentation, and classification of kidney tumours from CT scans. The system is designed to support radiologists in resource-constrained healthcare settings by reducing manual workload and improving diagnostic consistency.

### Pipeline Summary

| Stage | Model | Task | Target Metric |
|-------|-------|------|---------------|
| 1 | YOLOv8 | Kidney region detection | Sensitivity >90% & Precision >85% |
| 2 | 3D U-Net + ResNet50 | Tumour segmentation | Dice ≥0.80 & IoU ≥0.75 |
| 3 | EfficientNet-B0 | Benign/malignant classification | Accuracy >85% |
| Post | SHAP | Explainability | Visual attribution maps |

## Technical Stack
- Python 3.11.9
- PyTorch 2.0+
- Ultralytics YOLOv8
- segmentation_models_pytorch
- MONAI
- EfficientNet-B0 (ImageNet pretrained, fine-tuned on KiTS21)
- SHAP


### Dataset
- **KiTS21** (Kidney Tumour Segmentation Challenge 2021)
- 300 anonymized pre-operative CT cases
- NIfTI format, 512×512 pixels per slice
- Patient-level split: 110 (detection) / 120 (segmentation) / 
  70 (testing)

**Hardware:**
- Local: Intel Core i7 (11th gen), 8GB RAM — code writing and testing
- Cloud: Google Colab Pro+ with NVIDIA A100 (40GB VRAM) — training

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
├── configs/
│   └── config.yaml          # All paths and hyperparameters
│
├── outputs/                 # Gitignored - results saved to Drive
│   ├── logs/
│   ├── metrics/
│   └── checkpoints/
│
├── tests/                   # Module verification scripts
├── requirements-local.txt   # Local development dependencies
└── requirements-colab.txt   # Colab training dependencies
```
---

## Local Development Setup

These steps are intended to set up local machine for preprocessing and code editing only. Model training happens on Google Colab.

### Prerequisites
- Windows 10/11
- Python 3.11.x
- Git

### Steps

### Phase 1: Local Setup
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

### Phase 2: Google Colab + Drive Setup
Sets up the cloud environment where all model training runs.

**1. Activate Google One Premium**

**2. Open the Colab notebook**
```
notebooks/kidney_tumour_pipeline.ipynb
```
**3. At the start of every new Colab session, run Cell 0 first**
```python
# Cell 0 handles: Drive mount, kits21 clone,
# package install, and TRAINING_DIR redirect
# Takes approximately 2-3 minutes
```
---

### Phase 3: Dataset Download and Verification

Downloads and verifies all 300 KiTS21 cases to Google Drive.

**1. Clone KiTS21 repository to Colab**
```bash
git clone https://github.com/neheller/kits21.git /content/kits21
```

**2. Redirect TRAINING_DIR to Google Drive**
```python
# Handled automatically by Cell 0
```

**3. Download all 300 cases (~30GB)**
```python
from kits21.api.data import fetch_training_data
fetch_training_data()
```

**4. Verify dataset on Drive**
```
dataset/raw/case_00000/ through case_00299/
Each case contains: imaging.nii.gz + segmentation.nii.gz
```

---

## Implementation Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Local VS Code Setup | ✅ Complete |
| 2 | Google Colab + Drive Setup | ✅ Complete |
| 3 | Dataset Download and Verification | ✅ Complete |
| 4 | Preprocessing Pipeline | 🔄 In Progress |
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
