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
- Patient-level split: 110 (detection) / 120 (segmentation) / 70 (testing)

**Hardware:**
- Local: Intel Core i7 (11th gen), 8GB RAM — code writing and testing
- Cloud: Google Colab Pro+ with NVIDIA A100 (40GB VRAM) — training

---

## Repository Structure
```
kidney-tumour-detection/
│
├── notebooks/                          # Google Colab training notebooks
│   ├── kidney_tumour_pipeline.ipynb    # Session setup and data download
│   ├── 01_preprocessing.ipynb          # Phase 4 — data preparation and preprocessing
│   ├── 02_yolo_training.ipynb          # Phase 5 — YOLOv8 detection training
│   ├── 03_unet_training.ipynb          # Phase 6 — U-Net segmentation training
│   ├── 04_efficientnet_training.ipynb  # Phase 7 — EfficientNet classification training
│   └── 05_evaluation.ipynb             # Phase 10 — end-to-end evaluation and metrics
│
├── src/                                # Source code
│   ├── preprocessing/                  # NIfTI loading, slicing, label generation
│   │   ├── data_exploration.py         # Step 4.1 — dataset audit and statistics
│   │   ├── data_splitting.py           # Step 4.2 — patient-level case splitting
│   │   ├── slice_extraction.py         # Step 4.3 — NIfTI to PNG conversion
│   │   ├── yolo_label_generation.py    # Step 4.4 — YOLO bounding box labels
│   │   └── yolo_dataset_structure.py   # Step 4.5 — train.txt, val.txt, data.yaml
│   ├── detection/                      # YOLOv8 training
│   ├── segmentation/                   # U-Net training
│   ├── classification/                 # EfficientNet training
│   ├── explainability/                 # SHAP integration
│   └── evaluation/                     # Metrics and reporting
│
├── configs/
│   └── config.yaml                     # All paths and hyperparameters
│
├── outputs/                            # Gitignored — results saved to Drive
│   ├── logs/
│   ├── metrics/
│   └── checkpoints/
│
├── requirements-local.txt              # Local development dependencies
└── requirements-colab.txt              # Colab training dependencies
```
---

## Local Development Setup

These steps set up your local machine for preprocessing and code editing only. Model training happens on Google Colab.

### Prerequisites
- Windows 10/11
- Python 3.11.9
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

**2. Open the session setup notebook first — every session**
```
notebooks/kidney_tumour_pipeline.ipynb
```
**3. Run Cell 0 at the start of every new Colab session**
```python
# Drive mount, repo pull, package install
# Takes approximately 2-3 minutes
# Must be run before any other notebook
```

**4. Then open the relevant phase notebook**
```
notebooks/01_preprocessing.ipynb   ← Phase 4
notebooks/02_yolo_training.ipynb   ← Phase 5
notebooks/03_unet_training.ipynb   ← Phase 6
...
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
| 4 | Preprocessing Pipeline | ✅ Complete |
| 5 | YOLOv8 Detection Training | ⏳ Pending |
| 6 | U-Net Segmentation Training | ⏳ Pending |
| 7 | EfficientNet Classification Training | ⏳ Pending |
| 8 | SHAP Integration | ⏳ Pending |
| 9 | End-to-End Pipeline Integration | ⏳ Pending |
| 10 | Evaluation and Metrics | ⏳ Pending |

### Phase 4 — Preprocessing Pipeline (Complete)

| Step | Script | Description |
|------|--------|-------------|
| 4.1 | data_exploration.py | Dataset audit — integrity, intensity stats, label distribution, histology counts |
| 4.2 | data_splitting.py | Patient-level split — 110 detection / 120 segmentation / 70 test |
| 4.3 | slice_extraction.py | NIfTI volumes → PNG slices (512×512), masks scaled to 0/85/170/255 |
| 4.4 | yolo_label_generation.py | YOLO bounding box labels — 56,604 slices, 21,814 boxes across 110 cases |
| 4.5 | yolo_dataset_structure.py | train.txt, val.txt, yolo_data.yaml — stratified 100/10 split |

---

## Academic Context

This system is a **proof of concept** and is not intended for clinical deployment. All experiments use the publicly available, pre-anonymized KiTS21 dataset. Results will be reported in Chapter 4 of the project.

This work contributes to:
- **SDG 3** — Good Health and Well-being
- **SDG 9** — Industry, Innovation and Infrastructure