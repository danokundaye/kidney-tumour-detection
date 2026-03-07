# Kidney Tumour Detection — Web Demo

Interactive demonstration of the three-stage YOLO-UNet pipeline from the thesis:
**"Design and Implementation of a Hybrid YOLO-UNet Architecture for Automated Detection,
Segmentation, and Classification of Kidney Tumours in CT Scans"**

Author: Daniel Osaseri Okundaye — Miva Open University, 2026

---

## What this demo does

Runs a CT scan through the full three-stage pipeline and displays results in a clinical-style interface:

- **Stage 1** — YOLOv8 detects kidney regions per slice and draws bounding boxes
- **Stage 2** — ResNet-UNet segments predicted tumour boundaries
- **Stage 3** — EfficientNet-B0 classifies the lesion as benign or malignant
- **Stage 4** — Gradient saliency map showing which image regions drove the classification (optional, on demand)

Two modes of operation:
- **Real inference** — model weights present, runs actual pipeline on the uploaded scan
- **Placeholder** — weights absent, simulates pipeline using thesis metrics for UI demonstration

---

## Project structure

```
web/
├── backend/
│   ├── app.py                  ← Flask API server
│   ├── requirements.txt        ← Python dependencies
│   ├── weights/                ← not in repo, bundle on server (see Deployment)
│   │   ├── yolo_best.pt
│   │   ├── unet_best.pt
│   │   └── efnet_best.pt
│   ├── test_cases/             ← not in repo, bundle on server (see Deployment)
│   │   ├── case_00088/imaging.nii.gz
│   │   ├── case_00293/imaging.nii.gz
│   │   ├── case_00292/imaging.nii.gz
│   │   ├── case_00024/imaging.nii.gz
│   │   └── case_00001/imaging.nii.gz
│   └── demo_uploads/           ← created at runtime, gitignored
│
└── frontend/
    └── src/
        └── App.jsx             ← React single-file frontend
```

---

## Running locally

### Backend

**Minimum requirements (placeholder mode)**
```
Python 3.11+
flask, flask-cors, nibabel, numpy, Pillow
```

**Additional requirements (real inference mode)**
```
torch, torchvision, ultralytics
segmentation-models-pytorch, opencv-python
```

**Steps**

1. Navigate to the backend folder:
```bash
cd web/backend
```

2. Install dependencies:
```bash
pip install flask flask-cors nibabel numpy Pillow
```

3. Start the server:
```bash
python app.py
```

The server starts at `http://localhost:5000`. Startup output confirms the mode:
```
Loading models...
  All models loaded — real inference mode active.     ← weights found
  Placeholder mode — torch not installed.             ← torch missing
  Placeholder mode — missing weights: yolo, unet...  ← weights missing

Checking pre-loaded test cases...
  case_00088 (Malignant  ) — 512 slices
  ...

Mode    : real inference
Address : http://localhost:5000
```

### Frontend

The frontend is a single React file (`App.jsx`). Run it with Vite:

```bash
cd web/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in Chrome, Edge, or Brave (not Firefox — folder uploads require webkit directory support).

If the API address changes (different port or deployed URL), update line 3 of `App.jsx`:
```js
const API = "http://localhost:5000";
```

---

## Supported upload formats

| Format | How to upload | Notes |
|--------|--------------|-------|
| NIfTI file | Click "Browse NIfTI file", navigate into a case folder, select `imaging.nii.gz` | Standard KiTS21 format |
| DICOM folder | Click "DICOM Folder", select the folder containing `.dcm` files | Requires `dcm2niix.exe` next to `app.py` on Windows |
| PNG slices | Click "PNG Slices Folder", select the `images/` subfolder from `processed/slices/test/case_XXXXX/` | Preprocessing stage is skipped — slices already windowed |

CT windowing applied automatically for NIfTI and DICOM: clip to `[-79, 304]` HU, normalise to `[0, 255]`.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/cases` | List the 5 pre-loaded test cases with metadata |
| POST | `/run/<case_id>` | Run pipeline on a pre-loaded case |
| POST | `/upload` | Run pipeline on an uploaded scan (multipart, field name `scan`) |
| GET | `/status/<job_id>` | Poll job progress |
| POST | `/cancel/<job_id>` | Cancel a running job |
| POST | `/shap/<job_id>` | Trigger SHAP analysis (pipeline must be complete first) |
| GET | `/image/<job_id>/<stage>` | Fetch stage visualisation PNG (`stage1` through `stage4`) |
| GET | `/health` | Health check — returns mode and device |

---

## Pre-loaded test cases

Five cases from the KiTS21 raw dataset. Raw NIfTI files are used directly so the full pipeline including preprocessing runs on each case in the demo:

| Case | Label | Dice (3D) | Notes |
|------|-------|-----------|-------|
| case_00088 | Malignant | 0.682 | Best segmentation result across 70 test cases |
| case_00293 | Malignant | 0.218 | Second strongest result |
| case_00292 | Malignant | 0.106 | Mid-range result |
| case_00024 | Benign | 0.000 | One of two benign cases — reflects 27:1 class imbalance |
| case_00001 | Malignant | 0.000 | Zero Dice — demonstrates honest system limitations |

The NIfTI files for these cases are not in the repository. See Deployment below.

---

## Deployment

This section covers deploying the backend to a server and bundling the data files.

### Files to bundle on the server (not in repo)

After deploying, copy these directly to the server via `scp` or your hosting provider's file manager:

**Model weights** (from Google Drive `results/`)
```
results/phase5_yolo_retrain/yolov8s_retrain_run1/weights/best.pt  →  web/backend/weights/yolo_best.pt
results/phase6_unet/weights/best.pt                                →  web/backend/weights/unet_best.pt
results/phase7_efficientnet/weights/best.pt                        →  web/backend/weights/efnet_best.pt
```

**Test case NIfTI files** (from Google Drive `kits21/dataset/raw/`)
```
kits21/dataset/raw/case_00088/imaging.nii.gz  →  web/backend/test_cases/case_00088/imaging.nii.gz
kits21/dataset/raw/case_00293/imaging.nii.gz  →  web/backend/test_cases/case_00293/imaging.nii.gz
kits21/dataset/raw/case_00292/imaging.nii.gz  →  web/backend/test_cases/case_00292/imaging.nii.gz
kits21/dataset/raw/case_00024/imaging.nii.gz  →  web/backend/test_cases/case_00024/imaging.nii.gz
kits21/dataset/raw/case_00001/imaging.nii.gz  →  web/backend/test_cases/case_00001/imaging.nii.gz
```

These files are excluded from the repository via `.gitignore`. They must be bundled manually on any server that needs real inference mode.

### Why these are not in the repository

- Model weights are each 20–130MB, exceeding GitHub's 100MB per-file limit
- NIfTI files are each 50–300MB for the same reason
- Both sets of files can be reproduced from the training scripts in `src/` given the KiTS21 dataset

---

## Performance context

Results are from a 70-case held-out test set. The system was trained on 300 cases total.

| Metric | Value |
|--------|-------|
| YOLO mAP@0.5 | 0.534 |
| YOLO Precision | 0.643 |
| U-Net Mean Dice | 0.019 |
| U-Net Max Dice | 0.682 |
| EfficientNet Accuracy | 0.886 |
| EfficientNet AUC | 0.577 |

The segmentation Dice score reflects the limited 120-case training set. The architectural approach is validated — the performance ceiling is a dataset scale constraint, not a model design flaw.

**This system is a research proof-of-concept and is not validated for clinical use. All outputs must be reviewed by a qualified radiologist.**

---

## Updating the pre-loaded cases

To swap in different cases, edit `PRELOADED_CASES` in `app.py`:

```python
{
    "case_id":     "case_00042",
    "label":       "Malignant",       # or "Benign"
    "slice_count": None,              # auto-populated at startup from NIfTI header
    "dice_3d":     0.312,             # from predictions.csv
    "description": "Your description here.",
    "nifti_path":  "test_cases/case_00042/imaging.nii.gz",
},
```
And place the corresponding `imaging.nii.gz` in `test_cases/case_00042/` and restart the server.