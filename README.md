# Hybrid YOLO-UNet Architecture for Kidney Tumour Detection, Segmentation, and Classification

**Author:** Daniel Osaseri Okundaye  
**Institution:** Miva Open University, Abuja, Nigeria  
**Supervisor:** Isiekwene Chinyere Chioma  
**Degree:** B.Sc. (Hons) Computer Science  
**Year:** 2026


## Project Overview

This project implements a three-stage deep learning pipeline for automated detection, segmentation, and classification of kidney tumours from CT scans. The system is designed to support radiologists in resource-constrained healthcare settings by reducing manual workload and improving diagnostic consistency.

### Pipeline Summary

| Stage | Model | Task | Target Metric | Achieved |
|-------|-------|------|---------------|----------|
| 1 | YOLOv8 | Kidney region detection | Sensitivity >90% & Precision >85% | mAP@0.5: 0.534, Precision: 0.643 |
| 2 | U-Net + ResNet50 | Tumour segmentation | Dice ≥0.80 & IoU ≥0.75 | Mean Dice: 0.019, Max: 0.682 |
| 3 | EfficientNet-B0 | Benign/malignant classification | Accuracy >85% | Accuracy: 0.886, AUC: 0.577 |
| Post | SHAP | Explainability | Visual attribution maps | Complete |

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
- NIfTI format, variable slice dimensions (width: 512 pixels, height: varies per case)
- Slices per case: 29–1,059 (average ~544)
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
│   ├── kidney_tumour_pipeline.ipynb    # Phases 2 & 3 — session setup and data download
│   ├── 01_preprocessing.ipynb          # Phase 4 — data preparation and preprocessing
│   ├── 02_yolo_training.ipynb          # Phase 5 — YOLOv8 detection training
│   ├── 03_unet_training.ipynb          # Phase 6 — U-Net segmentation training
│   ├── 04_efficientnet_training.ipynb  # Phase 7 — EfficientNet classification training
│   ├── 05_shap_analysis.ipynb          # Phase 8 — SHAP explainability
│   ├── 06_pipeline.ipynb               # Phase 9 — end-to-end pipeline integration
│   └── 07_evaluation.ipynb             # Phase 10 — evaluation and metrics
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
│   │   ├── unet_crop_preparation.py    # Step 6.2 — crop kidney regions from YOLO boxes
│   │   ├── unet_split.py               # Step 6.3 — patient-level train/val split
│   │   └── unet_train.py               # Step 6.4 — U-Net training with ResNet50 encoder
│   ├── classification/                 # EfficientNet training
│   │   ├── patch_preparation.py        # Step 7.1 — extract tumour patches from U-Net crops
│   │   ├── patch_split.py              # Step 7.2 — patient-level train/val split
│   │   ├── efficientnet_train.py       # Step 7.3 — EfficientNet-B0 training
│   │   └── efficientnet_eval.py        # Step 7.4 — evaluation on val set
│   ├── explainability/                 # SHAP integration
│   │   ├── shap_efficientnet.py        # Phase 8 — SHAP for EfficientNet on test patches
│   │   └── shap_combined.py            # Phase 8 — combined SHAP for all three models
│   └── pipeline/                       # End-to-end pipeline
│       └── pipeline.py                 # Phase 9 — YOLO → U-Net → EfficientNet inference
│
├── configs/
│   ├── config.yaml                     # All paths and hyperparameters
│   └── yolo_test_eval.yaml             # YOLO val config for test set evaluation
│
├── outputs/                            # Gitignored — results saved to Drive
│   ├── logs/
│   ├── metrics/
│   └── checkpoints/
│
├── requirements-colab.txt              # Colab training dependencies
└── requirements-local.txt              # Local development dependencies
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
notebooks/01_preprocessing.ipynb          ← Phase 4
notebooks/02_yolo_training.ipynb          ← Phase 5
notebooks/03_unet_training.ipynb          ← Phase 6
notebooks/04_efficientnet_training.ipynb  ← Phase 7
notebooks/05_shap_analysis.ipynb          ← Phase 8
notebooks/06_pipeline.ipynb               ← Phase 9
notebooks/07_evaluation.ipynb             ← Phase 10
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
| 6 | U-Net Segmentation Training | ✅ Complete |
| 7 | EfficientNet Classification Training | ✅ Complete |
| 8 | SHAP Integration | ✅ Complete |
| 9 | End-to-End Pipeline Integration | ✅ Complete |
| 10 | Evaluation and Metrics | ✅ Complete |

---

### Phase 4 — Preprocessing Pipeline (Complete)

| Step | Script | Description |
|------|--------|-------------|
| 4.1 | data_exploration.py | Dataset audit — integrity, intensity stats, label distribution, histology counts |
| 4.2 | data_splitting.py | Patient-level split — 110 detection / 120 segmentation / 70 test |
| 4.3 | slice_extraction.py | NIfTI volumes → PNG slices, masks scaled to 0/85/170/255 |
| 4.4 | yolo_label_generation.py | YOLO bounding box labels — 56,604 slices, 21,814 boxes across 110 cases |
| 4.5 | yolo_dataset_structure.py | train.txt, val.txt, yolo_data.yaml — stratified 100/10 split |

---

### Phase 5 — YOLOv8 Detection Training (Complete)

**Model:** YOLOv8s (11.1M parameters)  
**Training hardware:** NVIDIA A100 40GB, Google Colab Pro+  
**Total training runs:** 11 (runs 1–7 experimental, run 8 baseline, run 10 final, retrain run 1 final inference model)

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

#### Results (Run 10 — Detection Training)

| Metric | Value | Notes |
|--------|-------|-------|
| mAP@0.5 | 0.619 | On filtered validation set (246 large slices) |
| Precision | 0.775 | IoU-based |
| Recall | 0.497 | IoU-based — understates real performance (see note) |

> **Note on metrics:** IoU-based recall (0.497) understates real detection capability because KiTS21 ground truth boxes are drawn tightly around mask boundaries while the model predicts slightly larger generalised boxes. Raw detection rate is more representative of pipeline performance — U-Net only requires the kidney to be present within the crop, not perfectly bounded.

#### YOLO Retrain — Inference Model for Segmentation Handoff

After Run 10, YOLO was retrained without the large-slice filter to improve detection across all slice sizes for the segmentation pipeline handoff. This retrain was evaluated on the 120 segmentation training cases.

**Final checkpoint:** `results/phase5_yolo_retrain/yolov8s_retrain_run1/weights/best.pt`

| Metric | Value | Notes |
|--------|-------|-------|
| Slice-level detection | 73.6% | 17,178 / 23,342 slices detected |
| Case-level detection | 99.2% | 119 / 120 cases with at least one detection |
| Confidence threshold | 0.10 | Lower threshold recovers missed detections |

> **Note:** case_00152 had zero detections — ground truth bounding box fallback used for this case in the segmentation pipeline.

#### Formal Test Set Evaluation (70 held-out cases, 35,492 slices)

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Precision | 0.6426 | >85% | Not met |
| Recall | 0.5060 | >90% | Not met |
| mAP@0.5 | 0.5340 | >0.90 | Not met |
| mAP@0.50:0.95 | 0.2405 | — | — |
| Case-level detection | 68/70 | — | 97.1% of test cases |

> **Note on slice vs case level:** Slice-level recall (0.506) is lower than case-level detection (97.1%) because some cases have two kidneys — missing one kidney on a slice reduces recall without preventing the pipeline from processing the case. For the end-to-end pipeline, case-level detection is the operationally relevant metric.

#### Pipeline Handoff to U-Net
When passing detections to the segmentation stage:
1. Run YOLOv8 at conf=0.10 per slice
2. If no box predicted → skip slice (or use ground truth fallback for case_00152)
3. If multiple boxes predicted → take highest confidence box only
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

### Phase 6 — U-Net Segmentation Training (Complete)

**Model:** U-Net with ResNet50 encoder (ImageNet pretrained)  
**Final checkpoint:** `results/phase6_unet/weights/best.pt`  
**Training hardware:** NVIDIA A100 40GB, Google Colab Pro+  
**Total training runs:** 4 (runs 1–3 exploratory, run 4 final)

#### Training Configuration (Final — Run 4)
| Parameter | Value |
|-----------|-------|
| Encoder | ResNet50 (ImageNet pretrained) |
| Loss | Combined Dice + BCE |
| Optimizer | Adam, lr=0.0001 |
| Scheduler | Cosine annealing (eta_min=1e-6) |
| Max epochs | 150 |
| Early stopping patience | 30 |
| Batch size | 16 |
| Input size | 256×256 |
| Healthy:abnormal ratio | 2:1 |
| Train cases | 96 |
| Val cases | 24 |
| Epochs trained | 67 (early stopping triggered) |

#### Dataset Split
| Split | Cases | Abnormal slices | Healthy slices | Total slices |
|-------|-------|-----------------|----------------|--------------|
| Train | 96 | 3,368 | 6,736 | 10,104 |
| Val | 24 | 1,031 | 2,062 | 3,093 |

#### Results

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Best Val Dice | 0.4678 | ≥0.80 | Achieved at epoch 37 |
| Best Val IoU | 0.3980 | ≥0.75 | Achieved at epoch 37 |
| Early stopping | Epoch 67 | — | No improvement for 30 epochs |

#### Key Lessons Learned
- **Drive I/O bottleneck:** Reading 17,382 individual files from Drive caused training to hang. Resolved by building a single `unet_crops_index.csv` (120 directory reads instead of 17,382 file reads) and extracting a local zip archive at session start
- **Validation set size:** Initial 12-case validation set (only 3 tumour cases) produced highly unstable Dice metrics with swings of ±0.25 between epochs. Increased to 24 cases (6 tumour cases) for more reliable epoch-by-epoch comparison
- **Case-level vs slice-level labels:** Using case-level abnormality as a proxy for slice-level region type introduced contradictory training signal — slices labeled abnormal but with empty masks. Slice-level labeling from mask pixels was explored but reduced training data too aggressively (1,232 vs 3,847 abnormal slices). Final approach retained case-level labels, relying on Dice+BCE loss to handle empty masks naturally
- **Sampling ratio:** 3:1 and 2:1 healthy:abnormal ratios were evaluated. 2:1 produced slightly more stable training curves and was used in the final run
- **Val Dice ceiling ~0.47:** Consistent across all runs, attributed to limited dataset size (120 cases), 2D slice-based processing losing volumetric context, and high case-to-case variability in the small validation set. The 0.80 target, based on literature using larger datasets and 3D volumetric training, was not achieved within the scope of this proof-of-concept

---

### Phase 7 — EfficientNet Classification Training (Complete)

**Model:** EfficientNet-B0 (ImageNet pretrained, all layers fine-tuned)  
**Final checkpoint:** `results/phase7_efficientnet/weights/best.pt`  
**Training hardware:** NVIDIA A100 40GB, Google Colab Pro+

#### Patch Preparation (Steps 7.1–7.2)

Patches are 224×224 pixel crops of abnormal regions extracted from segmented CT slices. The classification task is patient-level binary classification — malignant (renal cell carcinoma) vs benign (non-cancerous solid tumour) — based on biopsy-confirmed histology labels from `kits.json`.

**Hybrid patch extraction strategy:**  
YOLO's 73.6% slice-level detection rate left 87 of 120 segmentation training cases with empty U-Net crops — the 20% bounding box margin was insufficient for tumours growing peripherally beyond the kidney boundary. To recover these cases, a hybrid approach was used:

- **33 cases with valid U-Net crops:** Run U-Net inference. If Dice ≥ 0.40 vs ground truth → extract patch from predicted mask. If Dice < 0.40 → fall back to ground truth mask.
- **87 cases with empty crops:** Extract directly from original slices using ground truth masks.

| Source | Patches | Percentage |
|--------|---------|------------|
| U-Net predicted mask (Dice ≥ 0.40) | 906 | 18.2% |
| Ground truth mask fallback | 4,081 | 81.8% |
| **Total** | **4,987** | **100%** |

**Dataset constraint — benign cases:**  
KiTS21 contains only 25 benign cases out of 300 (8.3%). After patch preparation, only 5 benign cases produced patches above the minimum size threshold (10×10 pixels), giving a 27:1 malignant-to-benign patch imbalance.

#### Patient-Level Split
| Split | Malignant cases | Benign cases | Malignant patches | Benign patches |
|-------|-----------------|--------------|-------------------|----------------|
| Train | 74 | 4 | 3,827 | 125 |
| Val | 19 | 1 | 983 | 52 |

> **Note:** Val benign set is a single case (case_00156, 52 patches). Benign val metrics should be interpreted with caution given this constraint.

#### Training Configuration (Final)
| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B0, all layers trainable |
| Loss | BCEWithLogitsLoss, benign class weight = 3.0 |
| Optimizer | Adam, lr=0.0001 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping patience | 10 epochs (on mean F1) |
| Max epochs | 50 |
| Batch size | 32 |
| Benign oversampling | 15× |
| Malignant patch cap | 375 (3:1 ratio before oversampling) |
| Best epoch | 6 |

#### Results (Final — Primary Experiment)

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Accuracy | 0.858 | >85% | Met |
| F1 Malignant | 0.922 | — | Strong |
| F1 Benign | 0.205 | — | Constrained by dataset |
| Sensitivity | 0.884 | >90% | Near target |
| Specificity | 0.365 | — | Limited by 5 benign cases |
| AUC | 0.829 | — | Genuine discriminative signal |

#### Ablation Experiments

Three configurations were compared to understand the effect of class weighting and data ratio:

| Experiment | Data ratio | Class weight | Accuracy | F1 Benign | Sensitivity | Specificity | AUC |
|------------|-----------|--------------|----------|-----------|-------------|-------------|-----|
| Primary (3:1, w=3.0) | 375 mal / 125 ben | 3.0 | 0.858 | 0.205 | 0.884 | 0.365 | **0.829** |
| Balanced (1:1, w=1.0) | 125 mal / 125 ben | 1.0 | 0.755 | 0.165 | 0.769 | 0.481 | 0.693 |
| Balanced + weighted (1:1, w=3.0) | 125 mal / 125 ben | 3.0 | 0.890 | 0.230 | 0.920 | 0.327 | 0.778 |

> **Key finding:** Class weight matters more than data ratio at this dataset size. The primary experiment achieves the best AUC (0.829) despite using more malignant training data — reducing malignant patches below what the benign data can support hurts generalisation. The AUC of 0.829 on a severely imbalanced val set confirms the architecture has learned genuine discriminative signal; reliable benign classification requires substantially more labelled benign cases than the 5 available in KiTS21.

#### Key Lessons Learned
- **Patient-level label limitation:** `malignant` in kits.json is a patient-level diagnosis, not per-lesion. Patches from the same patient are all labelled identically regardless of whether they contain tumour or cyst tissue — a known limitation requiring per-lesion pathology labels to resolve
- **5 benign cases is the ceiling:** Oversampling, class weighting, or data ratio adjustment cannot overcome the limited benign training cases. The model consistently collapsed to predicting all-malignant without the malignant patch cap
- **Malignant patch cap critical:** Without capping malignant at 375, the gradient signal from 3,827 malignant patches overwhelmed the 125 benign patches despite 15× oversampling and class weighting, causing F1 Benign to remain 0.0 across all epochs
- **Mean F1 as primary metric:** Accuracy alone is misleading at 19:1 val imbalance; a model predicting malignant for everything scores 95% accuracy. Mean F1 of malignant and benign F1 penalises the model for ignoring either class

---

### Phase 8 — SHAP Explainability (Complete)

**Scripts:** `src/explainability/shap_efficientnet.py`, `src/explainability/shap_combined.py`  
**Notebook:** `notebooks/05_shap_analysis.ipynb`  
**Hardware:** NVIDIA T4, Google Colab Pro+

#### Overview

SHAP (SHapley Additive exPlanations) was applied to provide pixel-level attribution maps showing which image regions influenced each model's decision. GradientExplainer was used instead of DeepExplainer — EfficientNet's SiLU activations and residual `+=` operations are incompatible with DeepExplainer's backward hooks.

#### EfficientNet SHAP (shap_efficientnet.py)

Applied to 50 test patches sampled from 6 cases selected by Dice score:

| Case | Dice | Patches sampled | Rationale |
|------|------|-----------------|-----------|
| case_00088 | 0.6820 | 10 | Best segmentation result |
| case_00293 | 0.2183 | 10 | Moderate segmentation |
| case_00292 | 0.1060 | 10 | Moderate segmentation |
| case_00146 | 0.1155 | 10 | Moderate segmentation |
| case_00049 | 0.0000 | 5 | Contrast — poor segmentation |
| case_00001 | 0.0000 | 5 | Contrast — poor segmentation |

Background baseline: 50 patches sampled from `efficientnet_train.csv`.

**Outputs:** `results/phase8_shap/efficientnet/`
- `visualizations/` — 50 × 3-column figures (original / SHAP magnitude / overlay)
- `shap_values/shap_values.npy` — raw attribution arrays (50, 3, 224, 224)
- `shap_per_patch.csv` — per-patch magnitude metrics
- `shap_case_summary.csv` — aggregated metrics per case

#### Combined SHAP (shap_combined.py)

Applied to all three models on two representative cases to show attribution at each pipeline stage:

| Case | Dice | Purpose |
|------|------|---------|
| case_00088 | 0.6820 | Best case — focused attribution expected |
| case_00001 | 0.0000 | Contrast case — diffuse/misdirected attribution |

Each figure shows 3 rows (YOLO / U-Net / EfficientNet) × 3 columns (original / SHAP magnitude / overlay).

**Outputs:** `results/phase8_shap/combined/`
- `case_00088_slice_0190_combined.png`
- `case_00001_slice_0370_combined.png`

#### Key Findings
- **YOLO:** Attribution concentrated on kidney region on best case — model attends to correct anatomical area for detection confidence
- **U-Net:** On case_00088, attribution spread across kidney interior matching the tumour region — confirms model responds to genuine tissue texture on good cases. On case_00001, diffuse attribution across entire kidney with no focal point — consistent with zero Dice
- **EfficientNet:** On both cases, attribution maps are mostly near-zero (predominantly blue) with sparse bright spots. Bright regions correspond to edges and texture artifacts rather than clear tumour tissue — consistent with model receiving fallback crops outside its training distribution on most cases
- **SHAP for U-Net and YOLO excluded from standalone analysis:** U-Net mean Dice of 0.0189 on test cases means near-zero predictions produce uninformative attribution maps. Combined figure captures sufficient model-level comparison for thesis purposes

---

### Phase 9 — End-to-End Pipeline Integration (Complete)

**Script:** `src/pipeline/pipeline.py`  
**Notebook:** `notebooks/06_pipeline.ipynb`  
**Hardware:** NVIDIA A100 40GB, Google Colab Pro+  
**Runtime:** ~3.7 hours for 70 test cases

#### Pipeline Architecture

The pipeline chains all three models sequentially on each CT case:

```
CT slices (native dimensions, width = 512px)
    → YOLO (conf=0.10) → bounding box (highest confidence, 20% expansion)
    → crop + resize to 256×256
    → U-Net → binary segmentation mask (threshold=0.5)
    → patch extraction from mask
    → resize to 224×224
    → EfficientNet → malignant probability (threshold=0.5)
    → aggregate predictions → save .nii.gz mask + predictions.csv
```

#### Patch Extraction — Three-Tier Fallback Strategy

| Tier | Condition | Method |
|------|-----------|--------|
| 1 (contour) | Predicted mask ≥100 pixels | Bounding box from np.any scan, 10% expansion, crop from 256×256 |
| 2 (small crop) | Predicted mask >0 but <100 pixels | Full 256×256 crop used |
| 3 (empty crop) | Predicted mask = 0 | Full 256×256 crop used |

> **Note:** Tiers 2 and 3 pass crops outside EfficientNet's training distribution. Predictions from these tiers are flagged as `low_confidence` in outputs.

#### Results (70 Test Cases)

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Mean Dice (3D) | 0.0189 | ≥0.80 | Not met — see segmentation notes |
| Median Dice | 0.0000 | — | 58/70 cases have Dice = 0 |
| Max Dice | 0.6820 | — | case_00088 only |
| Mean processing time | 191.3s | <240s | Met for 84% of cases |
| Cases predicted malignant | 68/70 | — | — |
| Cases low confidence | 69/70 | — | Majority of patches from fallback tiers |

#### Patch Method Breakdown (All 70 Cases)

| Method | Patches | Percentage |
|--------|---------|------------|
| Contour (tier 1) | 2,636 | 7.4% |
| Small crop (tier 2) | 1,038 | 2.9% |
| Empty crop (tier 3) | 31,818 | 89.7% |
| **Total** | **35,492** | **100%** |

#### Outputs

```
results/phase9_pipeline/
├── masks/          case_xxxxx_pred_mask.nii.gz (70 files)
├── patches/        case_xxxxx/slice_xxxx.png
└── predictions.csv
```

**predictions.csv columns:** `case_id, n_slices, n_detected, detection_rate, dice_3d, iou_3d, n_patches, mean_prob, pred_label, confidence_flag, patches_contour, patches_small, patches_empty, patches_no_det, processing_time_s`

#### Key Findings
- **U-Net generalisation failure:** Val Dice of 0.4678 during training collapsed to mean 0.0189 on test cases — severe overfitting to training distribution. 89.7% of all patches came from the empty crop fallback, meaning U-Net predicted zero tumour pixels on most slices
- **Classification unreliable:** 69/70 cases flagged low confidence. EfficientNet predicted malignant for 68/70 cases — not genuine classification but bias from receiving fallback crops outside training distribution
- **Processing time:** Mean 191.3s exceeds 120s original target. Revised target of 240s is met for 84% of cases. Outliers with 400–542s correspond to cases with unusually high slice counts
- **Pipeline architecture validated:** The three-stage design is functionally correct — the bottleneck is model performance on limited data, not the integration logic

---

### Phase 10 — Evaluation and Metrics (Complete)

**Notebook:** `notebooks/07_evaluation.ipynb`  
**Hardware:** NVIDIA A100 40GB (YOLO test evaluation), T4 (remaining cells)

#### Final Results Summary

**Detection (YOLOv8 — 70 test cases, 35,492 slices)**

| Metric | Value | Target | Met? |
|--------|-------|--------|------|
| Precision | 0.6426 | >85% | No |
| Recall | 0.5060 | >90% | No |
| mAP@0.5 | 0.5340 | >0.90 | No |
| mAP@0.50:0.95 | 0.2405 | — | — |
| Case-level detection | 68/70 (97.1%) | — | — |

**Segmentation (U-Net — 70 test cases)**

| Metric | Value | Target | Met? |
|--------|-------|--------|------|
| Mean Dice | 0.0189 | ≥0.80 | No |
| Median Dice | 0.0000 | — | — |
| Mean IoU | 0.0123 | ≥0.75 | No |
| Best case Dice | 0.6820 (case_00088) | — | — |
| Cases Dice = 0 | 58/70 | — | — |

**Classification (EfficientNet — 70 test cases)**

| Metric | Value | Target | Met? |
|--------|-------|--------|------|
| Accuracy | 0.8857 | >85% | Technically yes |
| Sensitivity | 0.9688 | >90% | Technically yes |
| Specificity | 0.0000 | — | — |
| F1 Score | 0.9394 | — | — |
| AUC | 0.5768 | — | — |

> **Important:** Classification accuracy and sensitivity targets are technically met but misleading. The confusion matrix shows all 6 benign test cases were predicted malignant — the model is not genuinely classifying, it is predicting malignant for everything. AUC of 0.577 (barely above random) confirms this. The root cause is 89.7% of patches coming from the empty crop fallback, meaning EfficientNet received kidney tissue rather than tumour tissue on most cases.

**Processing Time**

| Metric | Value | Target |
|--------|-------|--------|
| Mean time | 191.3s | <240s |
| Cases under 240s | 59/70 (84%) | — |
| Cases under 120s | 2/70 (3%) | — |

#### Outputs Saved

```
results/phase10_evaluation/
├── yolo_metrics.csv
├── segmentation_metrics.csv
├── classification_metrics.csv
├── processing_time_metrics.csv
├── distributions.png                    # Dice / time / patch method charts
├── confusion_matrix.png
├── training_curves.png                  # All three models
└── case_00088_segmentation_overlay.png  # GT vs predicted mask comparison
```

---

## Academic Context

This system is a **proof of concept** and is not intended for clinical deployment. All experiments use the publicly available, pre-anonymized KiTS21 dataset. Results will be reported in Chapter 4 of the project.

This work contributes to:
- **SDG 3** — Good Health and Well-being
- **SDG 9** — Industry, Innovation and Infrastructure