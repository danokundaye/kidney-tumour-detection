# Phase 6, Step 6.1 - YOLO Inference on Segmentation Train Cases
#
#   This script runs the trained YOLOv8 model (best.pt from Phase 5) on all
#   segmentation_train cases to generate bounding box coordinates.
#   These boxes will be used in Step 6.2 to crop kidney regions for U-Net training.
#
# Steps:
#   - Confidence threshold: 0.10 (same as Phase 5 val evaluation)
#   - Per slice: take only the highest-confidence box
#   - Apply 20% margin expansion using actual image dimensions
#   - Save pixel-space coordinates [x1, y1, x2, y2] to JSON per case
#   - Skip slices with no detection
# 
# Execution: Google Colab
#
# Output Structure:
#   .../processed/unet_boxes/case_00000.json
#
# Input:
#   Slices from:
#   .../processed/slices/segmentation_train/case_00000/images/
# =============================================================================

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Load config.yaml
def load_config(config_path: str) -> dict:
    """
    Load the config.yaml file
    All paths and settings come from here
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Expand the boxes
def expand_box(
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float,
        img_w: int, 
        img_h: int, 
        margin: float = 0.20
        ):
    """
    Expand a bounding box by a fixed margin on all sides.
    Clip box to image boundaries so we never go out of bounds.

    Args:
        x1, y1, x2, y2 : Box corners in pixel space
        img_w, img_h   : Actual image dimensions (width, height)
        margin         : Fractional expansion (0.20 = 20%)

    Returns:
        Expanded box [x1, y1, x2, y2] clipped to image bounds
    """
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * margin
    pad_y = box_h * margin

    x1_new = max(0,     x1 - pad_x)
    y1_new = max(0,     y1 - pad_y)
    x2_new = min(img_w, x2 + pad_x)
    y2_new = min(img_h, y2 + pad_y)

    return [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]


# Inference on all slices of one case
def run_inference_on_case(
        case_id: str,
        slices_dir: Path,
        model: YOLO,
        conf_thresh: float,
        margin: float
    ) -> dict:
    """
    Run YOLO inference on all slices for one case.

    Returns a dict mapping slice filename → expanded box [x1, y1, x2, y2].
    Only slices with at least one detection are included.
    """
    images_dir = slices_dir / "segmentation_train" / case_id / "images"

    if not images_dir.exists():
        print(f"  WARNING: images dir not found for {case_id}, skipping")
        return {}

    slice_paths = sorted(images_dir.glob("*.png"))

    if len(slice_paths) == 0:
        print(f"  WARNING: no slices found for {case_id}, skipping")
        return {}

    case_boxes = {}

    for slice_path in slice_paths:
        # Get actual image dimensions — do NOT assume 512x512
        # Images in segmentation_train may be 611x512 or other sizes
        with Image.open(slice_path) as img:
            img_w, img_h = img.size

        # Run inference
        results = model.predict(str(slice_path), conf = conf_thresh, verbose = False)

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # No detection on this slice — skip it
            continue

        # Take only the highest-confidence box
        # boxes.conf is a tensor of confidence scores
        conf_scores = boxes.conf.cpu().numpy()
        best_idx    = int(np.argmax(conf_scores))

        # boxes.xyxy gives pixel-space coordinates [x1, y1, x2, y2]
        best_box = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = best_box

        # Apply 20% margin expansion, clipped to image bounds
        expanded = expand_box(x1, y1, x2, y2, img_w, img_h, margin)

        # Store under the slice filename (not full path to keep JSON portable)
        case_boxes[slice_path.name] = expanded

    return case_boxes

# Inference loop for all cases
def run_inference(model_path: str,
                  splits_dir: Path,
                  slices_dir: Path,
                  output_dir: Path,
                  conf_thresh: float,
                  margin: float):
    """
    Run inference across all 120 segmentation_train cases.
    Saves one JSON file per case to output_dir.
    """
    # Load the trained YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Load segmentation_train case list
    seg_csv  = splits_dir / "segmentation_train.csv"
    cases_df = pd.read_csv(seg_csv)
    case_ids = cases_df['case_id'].tolist()

    print(f"Cases to process : {len(case_ids)}")
    print(f"Conf threshold   : {conf_thresh}")
    print(f"Margin           : {margin * 100:.0f}%")
    print(f"Output dir       : {output_dir}")
    print()

    output_dir.mkdir(parents = True, exist_ok = True)

    # Track statistics
    total_slices   = 0
    detected_slices = 0
    skipped_cases  = []

    for case_id in tqdm(case_ids, desc="Running YOLO inference..."):
        out_path = output_dir / f"{case_id}.json"

        # Skip if already done (allows resuming after interruption)
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            total_slices    += len(list(
                (slices_dir / "segmentation_train" / case_id / "images").glob("*.png")
            ))
            detected_slices += len(existing)
            continue

        case_boxes = run_inference_on_case(
            case_id    = case_id,
            slices_dir = slices_dir,
            model      = model,
            conf_thresh= conf_thresh,
            margin     = margin
        )

        # Count total slices for this case
        images_dir = slices_dir / "segmentation_train" / case_id / "images"
        n_slices   = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        total_slices    += n_slices
        detected_slices += len(case_boxes)

        if len(case_boxes) == 0:
            skipped_cases.append(case_id)

        # Save JSON for this case
        out_path.write_text(json.dumps(case_boxes, indent = 2))

    # Print summary
    print(f"Inference Summary")
    print(f"  Total slices         : {total_slices}")
    print(f"  Detected slices      : {detected_slices}")
    print(f"  Detection rate       : {detected_slices/total_slices*100:.1f}%")
    print(f"  Cases with 0 detects : {len(skipped_cases)}")

    if skipped_cases:
        print(f"\n  Cases with no detections:")
        for cid in skipped_cases:
            print(f"    {cid}")

    print(f"\nJSON files saved to: {output_dir}")


# Main
def main():
    config_path = "/content/kidney-tumour-detection/configs/config.yaml"
    config      = load_config(config_path)

    # Paths
    model_path  = "/content/drive/MyDrive/kidney-tumour-detection/results/phase5_yolo/yolov8s_run10/weights/best.pt"
    slices_dir  = Path(config['paths']['slices_dir'])
    splits_dir  = Path(config['paths']['splits_dir'])
    output_dir  = Path(config['paths']['processed_root']) / "unet_boxes"

    # Inference settings
    conf_thresh = 0.10   # Same threshold used in Phase 5 val evaluation
    margin      = config['preprocessing']['bbox_margin']  # 0.20

    print(" Step 6.1 — YOLO Inference")

    run_inference(
        model_path  = model_path,
        splits_dir  = splits_dir,
        slices_dir  = slices_dir,
        output_dir  = output_dir,
        conf_thresh = conf_thresh,
        margin      = margin
    )


if __name__ == "__main__":
    main()