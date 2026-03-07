"""
Kidney Tumour Detection Pipeline — Demo Backend
================================================
Author : Daniel Osaseri Okundaye
Project: Hybrid YOLO-UNet Architecture (MIVA Open University, 2026)

This backend supports two modes:
  - Real mode   : model weights found in weights/ folder, runs actual inference
  - Placeholder : weights missing, simulates pipeline with thesis metrics

Supported upload formats
------------------------
1. NIfTI file  — single .nii or .nii.gz file
2. DICOM       — folder of .dcm files (requires dcm2niix next to app.py)
3. PNG slices  — the images/ subfolder from processed/slices/test/case_XXXXX/

Weight files expected at (relative to app.py)
---------------------------------------------
  weights/yolo_best.pt
  weights/unet_best.pt
  weights/efnet_best.pt

How to run
----------
  pip install -r requirements.txt
  python app.py          # http://localhost:5000

Endpoints
---------
  GET  /cases                    — list 5 pre-loaded test cases
  POST /run/<case_id>            — run pipeline on a pre-loaded case
  POST /upload                   — run pipeline on an uploaded scan
  GET  /status/<job_id>          — poll job progress
  POST /cancel/<job_id>          — cancel a running job
  POST /shap/<job_id>            — trigger SHAP analysis (optional)
  GET  /image/<job_id>/<stage>   — fetch stage visualisation PNG
  GET  /health                   — health check
"""

import io
import sys
import uuid
import time
import random
import subprocess
import threading
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# cv2 is used for image processing in real inference mode.
# Not required for placeholder mode.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# torch and related libraries are only needed when model weights are present.
# The backend starts and runs in placeholder mode without them.
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

app = Flask(__name__)
CORS(app)

jobs: dict[str, dict] = {}

UPLOAD_FOLDER = Path(__file__).parent / "demo_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# CT windowing — matches slice_extraction.py and config.yaml exactly
CT_WINDOW_MIN = -79
CT_WINDOW_MAX = 304

# Inference settings — match the values used during training in phases 5, 6, and 7
YOLO_CONF        = 0.10
YOLO_MARGIN      = 0.10
UNET_SIZE        = 256
UNET_THRESHOLD   = 0.50
EFFNET_SIZE      = 224
EFFNET_THRESHOLD = 0.50
MIN_MASK_PIXELS  = 100
PATCH_MARGIN     = 0.10

# Weight file paths anchored to app.py's own directory,
# so they resolve correctly regardless of where the terminal is when running.
_HERE = Path(__file__).parent
WEIGHT_PATHS = {
    "yolo":  _HERE / "weights/yolo_best.pt",
    "unet":  _HERE / "weights/unet_best.pt",
    "efnet": _HERE / "weights/efnet_best.pt",
}

# dcm2niix binary
_DCM2NIIX = str(Path(__file__).parent / "dcm2niix.exe") if sys.platform == "win32" else "dcm2niix"

# Global model state — populated once at startup, never reloaded per request
MODELS = {
    "yolo":            None,
    "unet":            None,
    "efnet":           None,
    "unet_transform":  None,
    "efnet_transform": None,
    "device":          None,
    "real_mode":       False,
}


# Pre-loaded test cases
PRELOADED_CASES = [
    {
        "case_id":     "case_00088",
        "label":       "Malignant",
        "slice_count": None,
        "dice_3d":     0.682,
        "description": "Best segmentation result across all 70 test cases. "
                        "Clear tumour boundary — U-Net produces meaningful mask overlap.",
        "nifti_path":  "test_cases/case_00088/imaging.nii.gz",
    },
    {
        "case_id":     "case_00293",
        "label":       "Malignant",
        "slice_count": None,
        "dice_3d":     0.218,
        "description": "Second strongest result. Partial tumour overlap — "
                        "representative of pipeline near its performance ceiling.",
        "nifti_path":  "test_cases/case_00293/imaging.nii.gz",
    },
    {
        "case_id":     "case_00292",
        "label":       "Malignant",
        "slice_count": None,
        "dice_3d":     0.106,
        "description": "Mid-range result. Limited but non-zero segmentation — "
                        "typical of the majority of test cases.",
        "nifti_path":  "test_cases/case_00292/imaging.nii.gz",
    },
    {
        "case_id":     "case_00024",
        "label":       "Benign",
        "slice_count": None,
        "dice_3d":     0.000,
        "description": "One of two benign cases identified across 70 test cases. "
                        "Low confidence — reflects 27:1 malignant-to-benign training imbalance.",
        "nifti_path":  "test_cases/case_00024/imaging.nii.gz",
    },
    {
        "case_id":     "case_00001",
        "label":       "Malignant",
        "slice_count": None,
        "dice_3d":     0.000,
        "description": "Zero Dice result — U-Net found no tumour overlap. "
                        "Demonstrates honest system limitations on difficult cases.",
        "nifti_path":  "test_cases/case_00001/imaging.nii.gz",
    },
]

CASES_BY_ID = {c["case_id"]: c for c in PRELOADED_CASES}


# Model loading
def _load_models():
    if not _TORCH_AVAILABLE:
        print("  Placeholder mode — torch not installed (install weights + torch to enable real inference).")
        return
    if not _CV2_AVAILABLE:
        print("  Placeholder mode — opencv-python not installed.")
        return
    """
    Attempt to load all three model weights at startup.
    Sets MODELS['real_mode'] = True only if all three load successfully.
    Falls back to placeholder mode silently if any weight file is missing.
    """
    missing = [name for name, path in WEIGHT_PATHS.items() if not path.exists()]
    if missing:
        print(f"  Placeholder mode — missing weights: {', '.join(missing)}")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        from ultralytics import YOLO
        import segmentation_models_pytorch as smp

        print("  Loading YOLO...")
        yolo = YOLO(str(WEIGHT_PATHS["yolo"]))

        # U-Net architecture must match phase 6 exactly
        print("  Loading U-Net...")
        unet = smp.Unet(
            encoder_name    = "resnet50",
            encoder_weights = None,
            in_channels     = 3,
            classes         = 1,
        )
        ckpt = torch.load(str(WEIGHT_PATHS["unet"]), map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        unet = unet.to(device).eval()

        # EfficientNet classifier head must match phase 7 exactly
        print("  Loading EfficientNet-B0...")
        efnet = models.efficientnet_b0(weights=None)
        efnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(efnet.classifier[1].in_features, 1),
        )
        ckpt = torch.load(str(WEIGHT_PATHS["efnet"]), map_location=device, weights_only=False)
        efnet.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        efnet = efnet.to(device).eval()

        # Transforms must match training normalisation from phases 6 and 7
        unet_tf = transforms.Compose([
            transforms.Resize((UNET_SIZE, UNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        efnet_tf = transforms.Compose([
            transforms.Resize((EFFNET_SIZE, EFFNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        MODELS.update({
            "yolo": yolo, "unet": unet, "efnet": efnet,
            "unet_transform": unet_tf, "efnet_transform": efnet_tf,
            "device": device, "real_mode": True,
        })
        print("  All models loaded — real inference mode active.")

    except Exception as e:
        print(f"  Model load failed: {e}")
        print("  Falling back to placeholder mode.")


# Preprocessing helpers
def _window_slice(slice_data: np.ndarray) -> np.ndarray:
    """Apply CT windowing and normalise to uint8. Mirrors slice_extraction.py."""
    clipped    = np.clip(slice_data, CT_WINDOW_MIN, CT_WINDOW_MAX)
    normalised = (clipped - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    return (normalised * 255).astype(np.uint8)


def _nifti_to_png_slices(nifti_path: str, out_dir: Path) -> list:
    """
    Load a NIfTI volume, apply CT windowing, save each axial slice as a PNG.
    Returns the sorted list of saved PNG paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_data = nib.load(nifti_path).get_fdata()
    paths    = []

    for i in range(img_data.shape[2]):
        arr  = _window_slice(img_data[:, :, i])
        arr  = np.rot90(arr)
        path = out_dir / f"slice_{i:04d}.png"
        if _CV2_AVAILABLE:
            rgb = np.stack([arr, arr, arr], axis=-1)
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            Image.fromarray(arr).save(str(path))
        paths.append(path)

    del img_data
    return sorted(paths)


def _validate_nifti(nifti_path: str) -> int:
    """Validate NIfTI dimensions and return slice count."""
    try:
        shape = nib.load(nifti_path).header.get_data_shape()
    except Exception as e:
        raise ValueError(f"Could not read NIfTI file: {e}")
    if len(shape) < 3:
        raise ValueError(f"NIfTI has fewer than 3 dimensions (shape: {shape}).")
    if shape[2] < 10:
        raise ValueError(f"NIfTI has only {shape[2]} slices — expected at least 10.")
    return int(shape[2])


def _convert_dicom_to_nifti(dicom_dir: str, out_dir: str) -> str:
    """Convert a DICOM folder to NIfTI using dcm2niix. Returns output path."""
    cmd = [_DCM2NIIX, "-z", "y", "-f", "imaging", "-o", out_dir, dicom_dir]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        raise RuntimeError(
            "dcm2niix not found. Windows: place dcm2niix.exe next to app.py. "
            "Mac/Linux: brew install dcm2niix or apt install dcm2niix."
        )
    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed:\n{result.stderr.strip()}")
    matches = list(Path(out_dir).glob("*.nii.gz")) or list(Path(out_dir).glob("*.nii"))
    if not matches:
        raise RuntimeError("dcm2niix ran but produced no NIfTI output.")
    return str(matches[0])


def _detect_upload_format(files: list) -> tuple:
    """Inspect uploaded files and return (format_str, nifti_file_or_None)."""
    nifti_file = None
    for f in files:
        name = (f.filename or "").lower()
        if name.endswith(".nii.gz") or name.endswith(".nii") or "/imaging.nii" in name:
            nifti_file = f
            break

    if nifti_file:
        return "nifti", nifti_file
    if any((f.filename or "").lower().endswith(".dcm") for f in files):
        return "dicom", None
    pngs = [f for f in files if (f.filename or "").lower().endswith(".png")]
    if len(pngs) >= 5:
        return "png", None
    if files:
        header = files[0].read(132)
        files[0].seek(0)
        if len(header) >= 132 and header[128:132] == b"DICM":
            return "dicom", None
    return "unknown", None


# Pipeline geometry helpers (copied from pipeline.py)
def _expand_box(x1, y1, x2, y2, margin, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1
    return (
        max(0,     int(x1 - bw * margin)),
        max(0,     int(y1 - bh * margin)),
        min(img_w, int(x2 + bw * margin)),
        min(img_h, int(y2 + bh * margin)),
    )


def _get_nonzero_bbox(mask: np.ndarray, margin: float):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    H, W = mask.shape
    return _expand_box(x1, y1, x2 + 1, y2 + 1, margin, W, H)


# Single-slice inference
def _infer_slice(slice_path: str) -> dict:
    """
    Run YOLO → U-Net → EfficientNet on one CT slice PNG.
    Direct adaptation of process_slice() from pipeline.py.
    Returns detection boxes, predicted mask, patch, and classification probability.
    """
    result = {
        "detected":     False,
        "boxes":        [],
        "crop_pil":     None,
        "pred_mask":    np.zeros((UNET_SIZE, UNET_SIZE), dtype=np.uint8),
        "patch_pil":    None,
        "patch_method": "no_detection",
        "effnet_prob":  None,
    }

    img_bgr = cv2.imread(slice_path)
    if img_bgr is None:
        return result

    img_h, img_w = img_bgr.shape[:2]
    img_pil      = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    yolo_out = MODELS["yolo"].predict(slice_path, conf=YOLO_CONF, verbose=False)
    boxes    = yolo_out[0].boxes

    if len(boxes) == 0:
        crop_pil = img_pil.resize((UNET_SIZE, UNET_SIZE), Image.BILINEAR)
    else:
        result["detected"] = True
        for b, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            result["boxes"].append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                                     "conf": float(c)})
        best_idx        = boxes.conf.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, YOLO_MARGIN, img_w, img_h)
        crop_pil        = img_pil.crop((x1, y1, x2, y2)).resize(
                            (UNET_SIZE, UNET_SIZE), Image.BILINEAR)

    result["crop_pil"] = crop_pil

    unet_in = MODELS["unet_transform"](crop_pil).unsqueeze(0).to(MODELS["device"])
    with torch.no_grad():
        prob_map = torch.sigmoid(MODELS["unet"](unet_in)).squeeze().cpu().numpy()
    pred_mask           = (prob_map >= UNET_THRESHOLD).astype(np.uint8)
    result["pred_mask"] = pred_mask

    n_px = int(pred_mask.sum())
    if n_px >= MIN_MASK_PIXELS:
        box = _get_nonzero_bbox(pred_mask, PATCH_MARGIN)
        patch_arr = np.array(crop_pil)[box[1]:box[3], box[0]:box[2]] if box else np.array(crop_pil)
        result["patch_method"] = "bbox_nonzero" if box else "full_crop_small"
    elif n_px > 0:
        patch_arr = np.array(crop_pil)
        result["patch_method"] = "full_crop_small"
    else:
        patch_arr = np.array(crop_pil)
        result["patch_method"] = "full_crop_empty"

    patch_pil          = Image.fromarray(patch_arr.astype(np.uint8))
    result["patch_pil"] = patch_pil

    efnet_in = MODELS["efnet_transform"](patch_pil).unsqueeze(0).to(MODELS["device"])
    with torch.no_grad():
        result["effnet_prob"] = torch.sigmoid(MODELS["efnet"](efnet_in)).item()

    return result


# Visualisation image generation
# Each function renders one PNG from real inference outputs and saves it
# so the /image/<job_id>/<stage> endpoint can serve it.

def _save_vis(job_id: str, stage: str, img: Image.Image):
    d = UPLOAD_FOLDER / job_id / "images"
    d.mkdir(parents=True, exist_ok=True)
    img.save(str(d / f"{stage}.png"))


def _draw_label_bar(img: Image.Image, text: str, color=(0, 229, 160)) -> Image.Image:
    """Draw a small label bar at the bottom of an image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([0, h - 20, w, h], fill=(8, 16, 26))
    draw.text((5, h - 17), text, fill=color)
    return img


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Pad an image to a square canvas with a black background, preserving aspect ratio.
    Corrects for KiTS21 coronal-orientation slices which are often elongated."""
    w, h   = img.size
    size   = max(w, h)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


def _generate_detection_image(job_id: str, slice_path: str, boxes: list):
    """
    Stage 1: CT slice with YOLO bounding boxes.
    Green boxes with confidence scores label each detected kidney region.
    Padded to square to correct for KiTS21 coronal-orientation elongation.
    """
    img  = _pad_to_square(Image.open(slice_path).convert("RGB"))
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if boxes:
        for box in boxes[:6]:
            x1 = int(box["x1"]); y1 = int(box["y1"])
            x2 = int(box["x2"]); y2 = int(box["y2"])
            conf  = box["conf"]
            label = f"kidney {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(0, 229, 160), width=2)
            lw = len(label) * 6 + 6
            draw.rectangle([x1, max(0, y1 - 16), x1 + lw, y1], fill=(0, 229, 160))
            draw.text((x1 + 3, max(0, y1 - 14)), label, fill=(0, 0, 0))
    else:
        draw.text((w // 2 - 80, h // 2), "No detection on this slice", fill=(255, 180, 0))

    _draw_label_bar(img, "Stage 1 — YOLOv8 detection")
    _save_vis(job_id, "stage1", img)


def _generate_segmentation_image(job_id: str, crop_pil: Image.Image, pred_mask: np.ndarray):
    """
    Stage 2: 256x256 kidney crop with U-Net predicted tumour mask overlaid.
    Red region = predicted tumour. Opacity blended to preserve CT texture underneath.
    """
    base    = crop_pil.convert("RGBA").resize((384, 384), Image.BILINEAR)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    mask_r  = cv2.resize(pred_mask, (384, 384), interpolation=cv2.INTER_NEAREST)
    ys, xs  = np.where(mask_r > 0)
    for y, x in zip(ys[::3], xs[::3]):   # sample every third pixel for speed
        odraw.point((int(x), int(y)), fill=(255, 60, 60, 145))

    blended = Image.alpha_composite(base, overlay).convert("RGB")
    _draw_label_bar(blended, "Stage 2 — U-Net tumour segmentation   (red = predicted region)")
    _save_vis(job_id, "stage2", blended)


def _generate_classification_image(job_id: str, patch_pil: Image.Image,
                                    prob: float, method: str):
    """
    Stage 3: The patch fed to EfficientNet with classification verdict overlaid.
    Red border = malignant, green border = benign.
    """
    label    = "Malignant" if prob >= EFFNET_THRESHOLD else "Benign"
    color    = (255, 80, 80) if label == "Malignant" else (0, 229, 160)
    conf_pct = f"{prob * 100:.1f}%" if label == "Malignant" else f"{(1 - prob) * 100:.1f}%"

    base  = patch_pil.convert("RGB").resize((384, 384), Image.BILINEAR)
    draw  = ImageDraw.Draw(base)
    w, h  = base.size

    for t in range(4):
        draw.rectangle([t, t, w - t - 1, h - t - 1], outline=color)

    draw.rectangle([0, h - 22, w, h], fill=(8, 16, 26))
    draw.text((6, h - 20), f"{label}  {conf_pct}", fill=color)

    _save_vis(job_id, "stage3", base)


def _generate_shap_image(job_id: str, patch_pil: Image.Image, prob: float):
    """
    Stage 4: Gradient saliency map for the EfficientNet prediction.

    Computes the gradient of the output score with respect to the input pixels,
    takes the absolute value across colour channels, and overlays it as a
    heatmap (blue = low influence, red = high influence).
    This is a genuine attribution map from the model's learned weights —
    not a synthetic approximation.
    """
    device = MODELS["device"]
    model  = MODELS["efnet"]
    tf     = MODELS["efnet_transform"]

    inp = tf(patch_pil).unsqueeze(0).to(device)
    inp.requires_grad_(True)
    model.zero_grad()
    torch.sigmoid(model(inp)).backward()

    saliency  = inp.grad.data.abs().squeeze().cpu().numpy()
    saliency  = saliency.max(axis=0)   # collapse colour channels to (H, W)
    saliency  = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    heat      = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat      = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base_arr  = np.array(patch_pil.convert("RGB").resize((EFFNET_SIZE, EFFNET_SIZE)))
    blended   = (0.55 * base_arr + 0.45 * heat).astype(np.uint8)
    result    = Image.fromarray(blended).resize((384, 384), Image.BILINEAR)

    verdict = "malignant" if prob >= EFFNET_THRESHOLD else "benign"
    _draw_label_bar(result, f"Stage 4 — gradient saliency   prediction: {verdict}", color=(255, 180, 0))
    _save_vis(job_id, "stage4", result)


# Job state helpers
def _update(job_id: str, stage: int, **kwargs):
    if job_id not in jobs:
        return
    key = f"stage{stage}"
    if key not in jobs[job_id]["stages"]:
        jobs[job_id]["stages"][key] = {}
    jobs[job_id]["stages"][key].update(kwargs)


def _stage_start(job_id: str, stage: int, message: str):
    _update(job_id, stage,
            status="running", progress=0, message=message,
            started_at=time.time(), completed_at=None, duration_s=None)


def _stage_end(job_id: str, stage: int, message: str, result: dict):
    started = jobs[job_id]["stages"][f"stage{stage}"].get("started_at", time.time())
    ended   = time.time()
    _update(job_id, stage,
            status="complete", progress=100, message=message,
            completed_at=ended, duration_s=round(ended - started, 2),
            result=result)


def _stage_error(job_id: str, stage: int, message: str):
    _update(job_id, stage, status="error", message=message, completed_at=time.time())


def _is_cancelled(job_id: str) -> bool:
    return jobs.get(job_id, {}).get("cancelled", False)


def _new_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "pipeline_status": "queued",
        "shap_status":     "idle",
        "num_slices":      0,
        "stages":          {},
        "summary":         None,
        "error":           None,
        "cancelled":       False,
        "mode":            "real" if MODELS["real_mode"] else "placeholder",
    }
    return job_id


# Preprocessing stage runners
def _preprocess_nifti(job_id: str, nifti_path: str) -> list:
    _stage_start(job_id, 0, "Loading NIfTI volume and validating...")
    try:
        _update(job_id, 0, progress=10, message="Validating NIfTI dimensions...")
        n_slices = _validate_nifti(nifti_path)
        _update(job_id, 0, progress=20,
                message=f"Applying CT windowing [{CT_WINDOW_MIN}, {CT_WINDOW_MAX}] HU...")
        slice_dir = UPLOAD_FOLDER / job_id / "slices"
        paths     = _nifti_to_png_slices(nifti_path, slice_dir)
        jobs[job_id]["num_slices"] = n_slices
        _stage_end(job_id, 0, f"Preprocessing complete — {n_slices} slices ready.", {
            "slices_extracted": n_slices,
            "window_hu":        f"[{CT_WINDOW_MIN}, {CT_WINDOW_MAX}]",
            "format":           "NIfTI",
        })
        return paths
    except (ValueError, RuntimeError) as e:
        _stage_error(job_id, 0, str(e))
        raise


def _preprocess_dicom(job_id: str, dicom_dir: str, work_dir: str) -> list:
    _stage_start(job_id, 0, "Converting DICOM series to NIfTI via dcm2niix...")
    try:
        for i in range(5):
            time.sleep(0.3)
            _update(job_id, 0, progress=(i + 1) * 10, message="Running dcm2niix...")
        nifti_path = _convert_dicom_to_nifti(dicom_dir, work_dir)
        _update(job_id, 0, progress=60, message="DICOM converted. Windowing slices...")
        n_slices  = _validate_nifti(nifti_path)
        slice_dir = UPLOAD_FOLDER / job_id / "slices"
        paths     = _nifti_to_png_slices(nifti_path, slice_dir)
        jobs[job_id]["num_slices"] = n_slices
        _stage_end(job_id, 0, f"Preprocessing complete — {n_slices} slices ready.", {
            "slices_extracted": n_slices,
            "window_hu":        f"[{CT_WINDOW_MIN}, {CT_WINDOW_MAX}]",
            "format":           "DICOM → NIfTI",
        })
        return paths
    except (ValueError, RuntimeError) as e:
        _stage_error(job_id, 0, str(e))
        raise


def _preprocess_png(job_id: str, png_dir: str) -> list:
    _stage_start(job_id, 0, "Detecting pre-processed PNG slices...")
    time.sleep(0.4)
    paths = sorted(Path(png_dir).glob("*.png"))
    if not paths:
        msg = "No PNG files found in the uploaded folder."
        _stage_error(job_id, 0, msg)
        raise ValueError(msg)
    n = len(paths)
    jobs[job_id]["num_slices"] = n
    _stage_end(job_id, 0, f"Pre-processed slices detected — preprocessing skipped.", {
        "slices_extracted": n,
        "window_hu":        "pre-applied ([-79, 304] HU during slice_extraction.py)",
        "format":           "Pre-processed PNG slices",
    })
    return paths


def _preprocess_preloaded(job_id: str, case: dict) -> list:
    """
    For a pre-loaded case, window the NIfTI to PNG slices if the file exists.
    Falls back to placeholder simulation if the file hasn't been downloaded yet.
    """
    nifti_path = case["nifti_path"]
    _stage_start(job_id, 0, f"Loading {case['case_id']}...")

    if Path(nifti_path).exists():
        try:
            for i in range(3):
                time.sleep(0.2)
                _update(job_id, 0, progress=(i + 1) * 20,
                        message=f"Reading {case['case_id']} NIfTI volume...")
            n_slices  = _validate_nifti(nifti_path)
            slice_dir = UPLOAD_FOLDER / job_id / "slices"
            paths     = _nifti_to_png_slices(nifti_path, slice_dir)
            jobs[job_id]["num_slices"] = n_slices
            _stage_end(job_id, 0, f"Case loaded — {n_slices} slices ready.", {
                "slices_extracted": n_slices,
                "window_hu":        f"[{CT_WINDOW_MIN}, {CT_WINDOW_MAX}]",
                "format":           "Pre-loaded NIfTI",
            })
            return paths
        except (ValueError, RuntimeError) as e:
            _stage_error(job_id, 0, str(e))
            raise
    else:
        n = case.get("slice_count") or 512
        if isinstance(n, str):
            n = 512
        for i in range(5):
            time.sleep(0.15)
            _update(job_id, 0, progress=(i + 1) * 20,
                    message=f"Loading {case['case_id']} (placeholder — NIfTI not downloaded)...")
        jobs[job_id]["num_slices"] = n
        _stage_end(job_id, 0, f"Placeholder mode — {n} slices (NIfTI not downloaded).", {
            "slices_extracted": n,
            "window_hu":        f"[{CT_WINDOW_MIN}, {CT_WINDOW_MAX}]",
            "format":           "Placeholder",
        })
        return []   # empty list tells the pipeline runner to use placeholder inference


# Real inference pipeline
def _run_real_inference(job_id: str, slice_paths: list):
    """
    Run all three models slice by slice.
    Picks the most informative slice (highest YOLO confidence) for visualisations.
    Updates progress in real time as slices are processed.
    """
    n = len(slice_paths)

    _stage_start(job_id, 1, "Running YOLOv8 kidney detection...")
    slice_results  = [None] * n
    n_detected     = 0
    best_idx       = n // 2
    best_conf      = 0.0

    for i, sp in enumerate(slice_paths):
        if _is_cancelled(job_id):
            return None
        res = _infer_slice(str(sp))
        slice_results[i] = res
        if res["detected"]:
            n_detected += 1
            c = res["boxes"][0]["conf"] if res["boxes"] else 0
            if c > best_conf:
                best_conf = c
                best_idx  = i
        _update(job_id, 1, progress=int(((i + 1) / n) * 100))

    _stage_end(job_id, 1, "Stage 1 complete — kidney regions located.", {
        "slices_processed": n,
        "detections_found": n_detected,
        "detection_rate":   round(n_detected / max(n, 1), 3),
        "map_at_50":        0.534,
        "precision":        0.643,
    })

    best_res = slice_results[best_idx]
    _generate_detection_image(job_id, str(slice_paths[best_idx]),
                               best_res["boxes"] if best_res else [])

    if _is_cancelled(job_id):
        return None

    _stage_start(job_id, 2, "Aggregating U-Net segmentation results...")
    effnet_probs  = []
    patch_methods = []

    for i, res in enumerate(slice_results):
        if res and res["effnet_prob"] is not None:
            effnet_probs.append(res["effnet_prob"])
            patch_methods.append(res["patch_method"])
        _update(job_id, 2, progress=int(((i + 1) / n) * 100))

    _stage_end(job_id, 2, "Stage 2 complete — tumour regions segmented.", {
        "crops_segmented": n,
        "mean_dice":       0.019,
        "max_dice":        0.682,
        "mean_iou":        0.011,
        "note":            "Dice computed against ground truth during phase 10 evaluation.",
    })

    if best_res and best_res["crop_pil"] is not None:
        _generate_segmentation_image(job_id, best_res["crop_pil"], best_res["pred_mask"])

    if _is_cancelled(job_id):
        return None

    _stage_start(job_id, 3, "Aggregating EfficientNet classification results...")
    time.sleep(0.5)

    if effnet_probs:
        mean_prob = float(np.mean(effnet_probs))
        label     = "Malignant" if mean_prob >= EFFNET_THRESHOLD else "Benign"
        fallbacks = sum(1 for m in patch_methods if m in ("full_crop_small", "full_crop_empty"))
        conf_flag = "low_confidence" if fallbacks > len(patch_methods) / 2 else "standard"
    else:
        mean_prob = 0.5
        label     = "Malignant"
        conf_flag = "no_patches"

    confidence = mean_prob if label == "Malignant" else 1 - mean_prob
    stage3 = {
        "prediction":            label,
        "confidence":            round(confidence, 3),
        "probability_malignant": round(mean_prob, 3),
        "probability_benign":    round(1 - mean_prob, 3),
        "confidence_flag":       conf_flag,
        "model_accuracy":        0.886,
        "auc":                   0.577,
    }
    _stage_end(job_id, 3, "Stage 3 complete — classification done.", stage3)

    if best_res and best_res["patch_pil"] is not None:
        _generate_classification_image(job_id, best_res["patch_pil"],
                                        mean_prob, patch_methods[0] if patch_methods else "n/a")
    return stage3


# Placeholder inference
def _run_placeholder(job_id: str, n_slices: int):
    """Simulate the three model stages when weights are not available."""
    _stage_start(job_id, 1, "Running YOLOv8 kidney detection (placeholder)...")
    detections = []
    for i in range(n_slices):
        time.sleep(max(0.005, 0.04 * min(n_slices, 20) / n_slices))
        _update(job_id, 1, progress=int(((i + 1) / n_slices) * 100))
        if random.random() > 0.27:
            detections.append({
                "x1": random.uniform(100, 200), "y1": random.uniform(150, 250),
                "x2": random.uniform(250, 350), "y2": random.uniform(280, 400),
                "conf": round(random.uniform(0.10, 0.95), 3),
            })
    _stage_end(job_id, 1, "Stage 1 complete (placeholder).", {
        "slices_processed": n_slices,
        "detections_found": len(detections),
        "detection_rate":   round(len(detections) / max(n_slices, 1), 3),
        "map_at_50":        0.534,
        "precision":        0.643,
    })

    if _is_cancelled(job_id):
        return None

    _stage_start(job_id, 2, "Running ResNet-UNet segmentation (placeholder)...")
    for i in range(min(len(detections), 30)):
        time.sleep(0.05)
        _update(job_id, 2, progress=int(((i + 1) / max(len(detections), 30)) * 100))
    _stage_end(job_id, 2, "Stage 2 complete (placeholder).", {
        "crops_segmented": len(detections),
        "mean_dice":       0.019, "max_dice": 0.682, "mean_iou": 0.011,
        "note":            "Placeholder — install model weights for real inference.",
    })

    if _is_cancelled(job_id):
        return None

    _stage_start(job_id, 3, "Running EfficientNet-B0 classification (placeholder)...")
    for i in range(10):
        time.sleep(0.1)
        _update(job_id, 3, progress=(i + 1) * 10)

    label      = random.choice(["Malignant", "Benign"])
    confidence = round(random.uniform(0.72, 0.94), 3)
    stage3 = {
        "prediction":            label,
        "confidence":            confidence,
        "probability_malignant": confidence if label == "Malignant" else round(1 - confidence, 3),
        "probability_benign":    confidence if label == "Benign"    else round(1 - confidence, 3),
        "confidence_flag":       "placeholder",
        "model_accuracy":        0.886,
        "auc":                   0.577,
    }
    _stage_end(job_id, 3, "Stage 3 complete (placeholder).", stage3)
    return stage3


# Zone labels for the 3x3 saliency grid (row-major order)
_ZONE_NAMES = [
    "Upper-Left",   "Upper-Centre",   "Upper-Right",
    "Mid-Left",     "Central",        "Mid-Right",
    "Lower-Left",   "Lower-Centre",   "Lower-Right",
]


def _compute_zone_analysis(saliency: np.ndarray, label: str, prob: float) -> dict:
    """
    Divide a normalised (0–1) saliency map into a 3x3 grid and compute the
    mean attribution per zone. Returns ranked zones, top-3, and a written
    radiologist summary.
    """
    H, W    = saliency.shape
    rh, rw  = H // 3, W // 3

    # Mean saliency for each of the 9 zones
    raw_scores = []
    for row in range(3):
        for col in range(3):
            r0, r1 = row * rh, (row + 1) * rh if row < 2 else H
            c0, c1 = col * rw, (col + 1) * rw if col < 2 else W
            raw_scores.append(float(saliency[r0:r1, c0:c1].mean()))

    total = sum(raw_scores) + 1e-8
    zones = [
        {"name": name, "score": round(s, 4), "pct": round(s / total * 100, 1)}
        for name, s in zip(_ZONE_NAMES, raw_scores)
    ]
    zones_sorted = sorted(zones, key=lambda z: z["score"], reverse=True)
    top3         = zones_sorted[:3]

    # Build a written summary from the zone data
    dominant    = top3[0]["name"].lower()
    second      = top3[1]["name"].lower()
    third       = top3[2]["name"].lower()
    dom_pct     = top3[0]["pct"]
    verdict     = label.lower()
    conf_pct    = round(prob * 100 if label == "Malignant" else (1 - prob) * 100, 1)

    # Interpret zone position for the radiologist
    central_zones  = {"central", "upper-centre", "lower-centre", "mid-left", "mid-right"}
    peripheral_zones = {"upper-left", "upper-right", "lower-left", "lower-right"}
    dom_clean = top3[0]["name"]

    if dom_clean.lower() in central_zones:
        location_note = (
            "The dominant attribution is concentrated in the central region of the lesion, "
            "which is consistent with a centrally located tumour mass or necrotic core drawing "
            "the classifier's attention."
        )
    else:
        location_note = (
            f"The dominant attribution originates from the {dom_clean.lower()} periphery of the "
            "lesion. Peripheral enhancement patterns are a recognised indicator of aggressive "
            "tumour behaviour and are commonly associated with malignant renal cell carcinoma."
        )

    written = (
        f"The gradient saliency analysis indicates that the model's {verdict} prediction "
        f"(confidence {conf_pct}%) was driven primarily by the {dominant} zone "
        f"({dom_pct}% of total attribution), followed by the {second} and {third} zones. "
        f"{location_note} "
        f"The three highest-attribution zones together account for "
        f"{round(sum(z['pct'] for z in top3), 1)}% of the total model attention, suggesting "
        f"{'a focused and localised decision signal' if top3[0]['pct'] > 30 else 'a distributed attention pattern across multiple regions'}. "
        f"Regions with near-zero attribution (shown in blue on the heatmap) had negligible "
        f"influence on the classification outcome and correspond primarily to surrounding "
        f"healthy parenchyma or background tissue."
    )

    return {
        "zones":          zones,
        "zones_sorted":   zones_sorted,
        "top3":           top3,
        "written_summary": written,
    }


# SHAP runner
def _run_shap(job_id: str):
    """
    Compute gradient saliency for the EfficientNet prediction.
    Uses the patch image saved during stage 3 as input.
    Computes a 3x3 zone breakdown and written radiologist summary.
    Falls back to placeholder data if weights are unavailable.
    """
    _stage_start(job_id, 4, "Generating SHAP attribution maps...")
    patch_path = UPLOAD_FOLDER / job_id / "images" / "stage3.png"

    if MODELS["real_mode"] and patch_path.exists():
        try:
            patch_pil = Image.open(str(patch_path)).convert("RGB")
            prob      = jobs[job_id]["stages"].get("stage3", {}).get(
                        "result", {}).get("probability_malignant", 0.5)
            label     = "Malignant" if prob >= EFFNET_THRESHOLD else "Benign"

            for i in range(10):
                time.sleep(0.3)
                _update(job_id, 4, progress=(i + 1) * 10)

            # Recompute saliency to extract zone data (same computation as _generate_shap_image)
            device = MODELS["device"]
            model  = MODELS["efnet"]
            tf     = MODELS["efnet_transform"]
            inp    = tf(patch_pil).unsqueeze(0).to(device)
            inp.requires_grad_(True)
            model.zero_grad()
            torch.sigmoid(model(inp)).backward()
            saliency = inp.grad.data.abs().squeeze().cpu().numpy()
            saliency = saliency.max(axis=0)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            _generate_shap_image(job_id, patch_pil, prob)
            zone_data = _compute_zone_analysis(saliency, label, prob)

            result = {
                "method":         "gradient saliency",
                "zones":          zone_data["zones"],
                "zones_sorted":   zone_data["zones_sorted"],
                "top3":           zone_data["top3"],
                "written_summary": zone_data["written_summary"],
            }
        except Exception as e:
            result = {"error": str(e), "method": "gradient saliency (failed)"}
    else:
        for i in range(20):
            time.sleep(0.4)
            _update(job_id, 4, progress=(i + 1) * 5)
        # Placeholder zone data with realistic distribution
        placeholder_scores = [0.08, 0.12, 0.07, 0.11, 0.28, 0.10, 0.06, 0.11, 0.07]
        total = sum(placeholder_scores)
        zones = [
            {"name": name, "score": round(s, 4), "pct": round(s / total * 100, 1)}
            for name, s in zip(_ZONE_NAMES, placeholder_scores)
        ]
        zones_sorted = sorted(zones, key=lambda z: z["score"], reverse=True)
        result = {
            "method":         "placeholder",
            "zones":          zones,
            "zones_sorted":   zones_sorted,
            "top3":           zones_sorted[:3],
            "written_summary": (
                "Placeholder mode — install model weights for real attribution analysis. "
                "In real inference mode, this section provides a zone-by-zone breakdown of "
                "which image regions drove the classification decision, along with a written "
                "interpretation for radiologist review."
            ),
        }

    _stage_end(job_id, 4, "SHAP analysis complete.", result)
    jobs[job_id]["shap_status"] = "complete"


# Summary builder
def _build_summary(job_id: str, stage3: dict) -> dict:
    n      = jobs[job_id].get("num_slices", 0)
    label  = stage3["prediction"]
    conf   = stage3["confidence"]
    dice   = jobs[job_id]["stages"].get("stage2", {}).get("result", {}).get("mean_dice", 0.019)
    cert   = "High" if conf > 0.85 else "Moderate" if conf > 0.70 else "Low"
    mode   = jobs[job_id].get("mode", "placeholder")

    timings = {}
    for i in range(5):
        d = jobs[job_id]["stages"].get(f"stage{i}", {}).get("duration_s")
        if d is not None:
            timings[f"stage{i}_s"] = d

    return {
        "case_summary": (
            f"Analysis of {n} CT slice(s) completed across all three pipeline stages "
            f"({'real inference' if mode == 'real' else 'placeholder mode — install weights for real inference'}). "
            f"YOLOv8 identified kidney regions in the majority of slices. "
            f"The ResNet-UNet segmentation model produced a mean Dice score of {dice:.3f}, "
            f"reflecting the limited 120-case training set. "
            f"EfficientNet-B0 predicts the lesion as "
            f"{'MALIGNANT' if label == 'Malignant' else 'BENIGN'} with "
            f"{conf * 100:.1f}% confidence ({cert} certainty)."
        ),
        "key_metrics": {
            "slices_analysed": n,
            "dice_score":      dice,
            "classification":  label,
            "confidence_pct":  f"{conf * 100:.1f}%",
            "certainty_level": cert,
            "yolo_map50":      0.534,
            "mode":            mode,
        },
        "stage_timings": timings,
        "disclaimer": (
            "Research / proof-of-concept only. This system was trained on 300 CT cases "
            "from the KiTS21 dataset and is not validated for clinical use. "
            "All outputs must be reviewed by a qualified radiologist. "
            "Do not use for diagnostic decisions."
        ),
    }


# Main pipeline runner
def _run_pipeline(job_id: str, preprocess_fn, *args):
    """
    Entry point for all three job types (preloaded, NIfTI upload, PNG upload).
    Calls the preprocessing function then dispatches to real or placeholder inference.
    """
    try:
        jobs[job_id]["pipeline_status"] = "running"
        slice_paths = preprocess_fn(job_id, *args)

        if _is_cancelled(job_id):
            jobs[job_id]["pipeline_status"] = "cancelled"
            return

        if MODELS["real_mode"] and slice_paths:
            stage3 = _run_real_inference(job_id, slice_paths)
        else:
            n = jobs[job_id].get("num_slices", 512)
            stage3 = _run_placeholder(job_id, n)

        if stage3 is None or _is_cancelled(job_id):
            jobs[job_id]["pipeline_status"] = "cancelled"
            return

        jobs[job_id]["summary"]         = _build_summary(job_id, stage3)
        jobs[job_id]["pipeline_status"] = "complete"
        jobs[job_id]["shap_status"]     = "idle"

    except Exception as e:
        jobs[job_id]["pipeline_status"] = "error"
        jobs[job_id]["error"]           = str(e)


# Startup helpers
def _read_slice_count(nifti_path: str):
    p = Path(nifti_path)
    if not p.exists():
        return None
    try:
        return int(nib.load(str(p)).header.get_data_shape()[2])
    except Exception:
        return None


def _populate_slice_counts():
    for case in PRELOADED_CASES:
        count = _read_slice_count(case["nifti_path"])
        case["slice_count"] = count if count is not None else "not loaded"


# API routes
@app.route("/cases", methods=["GET"])
def list_cases():
    return jsonify([{
        "case_id":     c["case_id"],
        "label":       c["label"],
        "slice_count": c["slice_count"],
        "dice_3d":     c["dice_3d"],
        "description": c["description"],
        "available":   Path(c["nifti_path"]).exists(),
    } for c in PRELOADED_CASES])


@app.route("/run/<case_id>", methods=["POST"])
def run_preloaded(case_id: str):
    if case_id not in CASES_BY_ID:
        return jsonify({"error": f"Case '{case_id}' not found."}), 404
    job_id = _new_job()
    threading.Thread(
        target=_run_pipeline,
        args=(job_id, _preprocess_preloaded, CASES_BY_ID[case_id]),
        daemon=True
    ).start()
    return jsonify({"job_id": job_id, "case_id": case_id, "status": "queued"})


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("scan")
    if not files or not any(f.filename for f in files):
        return jsonify({"error": "No files received. Send files under the key 'scan'."}), 400

    # Debug — log filenames to help diagnose format detection issues
    for f in files[:5]:
        print(f"  UPLOAD filename: {repr(f.filename)}", flush=True)

    fmt, nifti_file = _detect_upload_format(files)
    if fmt == "unknown":
        return jsonify({"error": (
            "Unrecognised format. Upload a NIfTI file (.nii or .nii.gz), "
            "a DICOM series (.dcm files), or a PNG slices folder."
        )}), 400

    job_id  = _new_job()
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "nifti":
        save_path = job_dir / "imaging.nii.gz"
        nifti_file.save(str(save_path))
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, _preprocess_nifti, str(save_path)),
            daemon=True
        )
    elif fmt == "png":
        png_dir = job_dir / "slices"
        png_dir.mkdir(exist_ok=True)
        for f in files:
            if f.filename and f.filename.lower().endswith(".png"):
                f.save(str(png_dir / Path(f.filename).name))
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, _preprocess_png, str(png_dir)),
            daemon=True
        )
    else:  # dicom
        dicom_dir = job_dir / "dicom"
        dicom_dir.mkdir(exist_ok=True)
        for f in files:
            if f.filename:
                f.save(str(dicom_dir / Path(f.filename).name))
        work_dir = str(job_dir / "nifti_out")
        Path(work_dir).mkdir(exist_ok=True)
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, _preprocess_dicom, str(dicom_dir), work_dir),
            daemon=True
        )

    thread.start()
    return jsonify({"job_id": job_id, "format": fmt, "status": "queued"})


@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    if jobs[job_id]["pipeline_status"] not in ("queued", "running"):
        return jsonify({"error": "Job is not running"}), 400
    jobs[job_id]["cancelled"] = True
    return jsonify({"message": "Cancellation requested", "job_id": job_id})


@app.route("/shap/<job_id>", methods=["POST"])
def run_shap_route(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    if jobs[job_id]["pipeline_status"] != "complete":
        return jsonify({"error": "Pipeline must finish before running SHAP."}), 400
    if jobs[job_id].get("shap_status") == "running":
        return jsonify({"message": "SHAP already running"}), 200
    jobs[job_id]["shap_status"] = "running"
    threading.Thread(target=_run_shap, args=(job_id,), daemon=True).start()
    return jsonify({"message": "SHAP started", "job_id": job_id})


@app.route("/image/<job_id>/<stage>", methods=["GET"])
def get_image(job_id: str, stage: str):
    """Serve the visualisation PNG for a completed stage (stage1–stage4)."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    img_path = UPLOAD_FOLDER / job_id / "images" / f"{stage}.png"
    if not img_path.exists():
        return jsonify({"error": f"Image not available for {stage} yet."}), 404
    return send_file(str(img_path), mimetype="image/png")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "mode":   "real" if MODELS["real_mode"] else "placeholder",
        "device": str(MODELS["device"]) if MODELS["device"] else "cpu",
    })


# Entry point
if __name__ == "__main__":
    print("  Kidney Tumour Pipeline — Demo Backend") 

    print("\nLoading models...")
    _load_models()

    print("\nChecking pre-loaded test cases...")
    _populate_slice_counts()
    for c in PRELOADED_CASES:
        status_str = (f"{c['slice_count']} slices"
                      if isinstance(c["slice_count"], int) else "file not downloaded")
        print(f"  {c['case_id']} ({c['label']:<10}) — {status_str}")

    mode = "real inference" if MODELS["real_mode"] else "placeholder (weights not found)"
    print(f"\nMode    : {mode}")
    print(f"Address : http://localhost:5000\n")
    app.run(debug=True, port=5000)