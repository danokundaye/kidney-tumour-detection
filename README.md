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

**2. Open the session setup notebook**
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
| 5 | YOLOv8 Detection Training | ✅ Complete |
| 6 | U-Net Segmentation Training | ⏳ Pending |
| 7 | EfficientNet Classification Training | ⏳ Pending |
| 8 | SHAP Integration | ⏳ Pending |
| 9 | End-to-End Pipeline Integration | ⏳ Pending |
| 10 | Evaluation and Metrics | ⏳ Pending |

---

### Phase 4 — Preprocessing Pipeline (Complete)

| Step | Script | Description |
|------|--------|-------------|
| 4.1 | data_exploration.py | Dataset audit — integrity, intensity stats, label distribution, histology counts |
| 4.2 | data_splitting.py | Patient-level split — 110 detection / 120 segmentation / 70 test |
| 4.3 | slice_extraction.py | NIfTI volumes → PNG slices (512×512), masks scaled to 0/85/170/255 |
| 4.4 | yolo_label_generation.py | YOLO bounding box labels — 56,604 slices, 21,814 boxes across 110 cases |
| 4.5 | yolo_dataset_structure.py | train.txt, val.txt, yolo_data.yaml — stratified 100/10 split |

---

### Phase 5 — YOLOv8 Detection Training (Complete)

**Model:** YOLOv8s (11.1M parameters)  
**Final checkpoint:** `results/phase5_yolo/yolov8s_run10/weights/best.pt`  
**Training hardware:** NVIDIA A100 40GB, Google Colab Pro+  
**Total training runs:** 10 (runs 1–7 experimental, run 8 baseline, run 10 final)

#### Training Configuration (Final — Run 10)
| Parameter | Value |
|-----------|-------|
| Epochs | 100 (patience=0, full run) |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Image size | 512×512 |
| Training images | 7,706 (filtered) |
| Confidence threshold (inference) | 0.10 |

#### Dataset Filtering Strategy
The original 51,484 training images contained 63% background slices (no kidney visible), which caused the model to plateau early across multiple runs. Both training and validation sets were filtered to match the same distribution:

- **Kept:** Large kidney slices only (bounding box height >10%, width >5% of image) — 3,853 train / 246 val
- **Kept:** 1:1 background ratio — 3,853 train / 246 val background slices
- **Discarded:** 14,942 boundary slices where kidney occupies less than 10% of image height
- **Final training set:** 7,706 images (50% kidney, 50% background)
- **Final validation set:** 492 images (50% kidney, 50% background)

> **Key insight:** Matching validation distribution to training distribution produced reliable mAP signals during training, enabling meaningful epoch-by-epoch monitoring and correct best checkpoint selection.

#### Results

| Metric | Value | Notes |
|--------|-------|-------|
| Raw detection rate | 94.3% | On large kidney slices at conf=0.10 |
| mAP@0.5 | 0.619 | On filtered validation set (246 large slices) |
| Precision | 0.775 | IoU-based |
| Recall | 0.497 | IoU-based — understates real performance (see note) |

> **Note on metrics:** IoU-based recall (0.497) understates real detection capability because KiTS21 ground truth boxes are drawn tightly around mask boundaries while the model predicts slightly larger generalised boxes. Raw detection rate (94.3%) is more representative of pipeline performance — U-Net only requires the kidney to be present within the crop, not perfectly bounded.

#### Pipeline Handoff to U-Net
When passing detections to the segmentation stage:
1. Run YOLOv8 at conf=0.10 per slice
2. If no box predicted → skip slice
3. If multiple boxes predicted → take highest confidence box only (NMS)
4. Expand selected box by 20% margin on all sides
5. Crop slice to expanded box → pass to U-Net

#### Key Lessons Learned
- **Background dominance (63%)** was the primary cause of poor detection across early runs
- **Matching train/val distribution** is critical — mismatched validation set produced unreliable mAP signals and caused premature early stopping
- **Early stopping unreliable** with mismatched val set — disabled with patience=0 for final runs
- **Large batch sizes (64–128)** caused over-averaged gradients and early plateaus — batch=16 optimal
- **cls=1.5 loss weight** made model overly conservative, destroying recall
- **Lower confidence threshold (0.10)** recovers missed detections without retraining — 70.7% → 94.3%
- **Multiple boxes per slice** appear at conf=0.10 — always take highest confidence box for U-Net

---

## Academic Context

This system is a **proof of concept** and is not intended for clinical deployment. All experiments use the publicly available, pre-anonymized KiTS21 dataset. Results will be reported in Chapter 4 of the project.

This work contributes to:
- **SDG 3** — Good Health and Well-being
- **SDG 9** — Industry, Innovation and Infrastructure