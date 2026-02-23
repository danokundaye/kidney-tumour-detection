# Step 6.7 — U-Net Inference Visualisation
#
# This scruot loads the trained U-Net (best.pt) and runs inference on all 24 val cases.
# Saves two 4-column grid images per case to Drive:
#
# Output:
#   Binary grid:     CT crop | GT (abnormal=white) | Prediction | Overlay
#   Multiclass grid: CT crop | GT (kidney=green, tumour=red, cyst=blue)
#                            | Prediction | Overlay


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Paths
DRIVE_ROOT  = Path("/content/drive/MyDrive/kidney-tumour-detection")
CROPS_DIR   = DRIVE_ROOT / "dataset/processed/unet_crops"
VAL_CSV     = DRIVE_ROOT / "dataset/processed/splits/unet_val.csv"
CHECKPOINT  = DRIVE_ROOT / "results/phase6_unet/weights/best.pt"
OUT_DIR     = DRIVE_ROOT / "results/phase6_unet/visualizations"

# Settings
MAX_SLICES  = 8       # Max slices to show per case
THRESHOLD   = 0.5     # Binary prediction threshold
IMG_SIZE    = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mask pixel values
KIDNEY_VAL, TUMOUR_VAL, CYST_VAL = 85, 170, 255


# Load model
def load_model():
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None
    )
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    # Handle both raw state dict and wrapped checkpoint formats
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"Model loaded from {CHECKPOINT.name}. Device: {DEVICE}")
    return model


# Per-slice helpers
def load_image(path):
    """
    Load PNG crop and normalize to match Phase 6 training preprocessing.
    Training used A.Normalize(mean=[0.485], std=[0.229]):
      output = (pixel/255 - mean) / std
    """
    img       = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_float = np.array(img, dtype=np.float32) / 255.0
    img_norm  = (img_float - 0.485) / 0.229
    return img_norm

def load_mask(path):
    """
    Load PNG mask preserving original pixel values (0,85,170,255).
    """
    mask = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return np.array(mask, dtype=np.uint8)

@torch.no_grad()
def infer(model, img_np):
    """
    Run U-Net inference. Returns binary prediction mask (H,W) uint8.
    """
    t = torch.from_numpy(
        np.stack([img_np, img_np, img_np])  # repeat grayscale → 3 channels
    ).unsqueeze(0).to(DEVICE)               # (1, 3, H, W)
    prob = torch.sigmoid(model(t))
    return (prob.squeeze().cpu().numpy() > THRESHOLD).astype(np.uint8)

def binary_gt(mask):
    """
    Collapse GT to binary: tumour+cyst pixels = 255, rest = 0.
    """
    out = np.zeros_like(mask)
    out[(mask == TUMOUR_VAL) | (mask == CYST_VAL)] = 255
    return out

def multiclass_gt(mask):
    """
    Convert GT to RGB: kidney=green, tumour=red, cyst=blue.
    """
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == KIDNEY_VAL] = [0,   200, 0  ]
    rgb[mask == TUMOUR_VAL] = [220, 0,   0  ]
    rgb[mask == CYST_VAL]   = [0,   100, 220]
    return rgb

def make_overlay(img_np, pred):
    """
    Rescale normalized image to [0,1] before converting to RGB for display.
    """
    img_display = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    rgb = (np.stack([img_display]*3, axis=-1) * 255).astype(np.uint8)
    rgb[pred == 1] = [220, 30, 30]
    return rgb

def compute_dice(pred, gt_bin):
    """
    Dice coefficient. Returns 1.0 if both masks are empty
    """
    g = (gt_bin > 0).astype(np.uint8)
    s = pred.sum() + g.sum()
    return float(2 * (pred * g).sum() / s) if s > 0 else 1.0


# Slice selection
def select_slices(case_dir):
    """
    Select up to MAX_SLICES slice stems per case.
    Priority: tumour slices → cyst slices → kidney-only slices.
    Samples evenly within each group for visual variety.
    """
    def sample_evenly(lst, n):
        if not lst or n == 0:
            return []
        idx = np.linspace(0, len(lst)-1, min(n, len(lst)), dtype=int)
        return [lst[i] for i in idx]

    tumour, cyst, kidney = [], [], []
    for f in sorted((case_dir / "masks").glob("*.png")):
        m = load_mask(f)
        if   np.any(m == TUMOUR_VAL): tumour.append(f.stem)
        elif np.any(m == CYST_VAL):   cyst.append(f.stem)
        elif np.any(m == KIDNEY_VAL): kidney.append(f.stem)

    selected = []
    for group in [tumour, cyst, kidney]:
        needed = MAX_SLICES - len(selected)
        if needed <= 0:
            break
        selected += sample_evenly(group, needed)
    return selected


# Grid generation
def save_grid(rows, case_id, mode, out_path, meta):
    """
    Save a (n_slices × 4) grid image for one case.
    rows: list of dicts — slice_name, img, gt, pred, overlay
    mode: 'binary' or 'multiclass'
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["CT Crop", "Ground Truth", "Prediction", "Overlay"]):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=8)

    for i, row in enumerate(rows):
        axes[i, 0].imshow(row['img'], cmap='gray')
        axes[i, 2].imshow(row['pred'],    cmap='gray', vmin=0, vmax=1)
        axes[i, 3].imshow(row['overlay'])
        axes[i, 0].set_ylabel(row['slice_name'], fontsize=8,
                               rotation=0, labelpad=65, va='center')

        if mode == 'binary':
            axes[i, 1].imshow(row['gt'], cmap='gray', vmin=0, vmax=255)
        else:
            axes[i, 1].imshow(row['gt'])  # RGB array, no cmap needed

        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

    if mode == 'multiclass':
        fig.legend(
            handles=[
                mpatches.Patch(color=(0, 200/255, 0),       label='Kidney'),
                mpatches.Patch(color=(220/255, 0, 0),       label='Tumour'),
                mpatches.Patch(color=(0, 100/255, 220/255), label='Cyst'),
            ],
            loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.0)
        )

    fig.suptitle(
        f"{case_id}  |  {meta['abnormality']}  |  {meta['histology']}  |  {mode}",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# Main
def main():
    print("Step 6.7 — U-Net Inference Visualisation\n")

    (OUT_DIR / "binary").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "multiclass").mkdir(parents=True, exist_ok=True)

    model  = load_model()
    val_df = pd.read_csv(VAL_CSV)
    metrics = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Cases"):
        case_id  = row['case_id']
        case_dir = CROPS_DIR / case_id
        meta     = {'abnormality': row['abnormality'], 'histology': row['histology']}

        if not case_dir.exists():
            print(f"  [skip] {case_id} — crops not found")
            continue

        stems = select_slices(case_dir)
        if not stems:
            continue

        rows_bin, rows_mc, dice_scores = [], [], []

        for stem in stems:
            img_np  = load_image(case_dir / "images" / f"{stem}.png")
            gt_mask = load_mask(case_dir  / "masks"  / f"{stem}.png")
            pred    = infer(model, img_np)
            gt_bin  = binary_gt(gt_mask)

            base = {
                'slice_name': stem,
                'img'       : img_np,
                'pred'      : pred,
                'overlay'   : make_overlay(img_np, pred),
            }
            rows_bin.append({**base, 'gt': gt_bin})
            rows_mc.append( {**base, 'gt': multiclass_gt(gt_mask)})
            dice_scores.append(compute_dice(pred, gt_bin))

        save_grid(rows_bin, case_id, 'binary',
                  OUT_DIR / "binary"     / f"{case_id}_binary.png",     meta)
        save_grid(rows_mc,  case_id, 'multiclass',
                  OUT_DIR / "multiclass" / f"{case_id}_multiclass.png", meta)

        metrics.append({
            'case_id'     : case_id,
            'abnormality' : meta['abnormality'],
            'histology'   : meta['histology'],
            'slices_shown': len(stems),
            'mean_dice'   : round(np.mean(dice_scores), 4),
        })

    df = pd.DataFrame(metrics)
    df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    print(f"\nCases visualised : {len(df)}")
    print(f"Overall mean Dice: {df['mean_dice'].mean():.4f}")
    print(f"Output saved to  : {OUT_DIR}")
    print("\nPer-case summary:")
    print(df[['case_id', 'abnormality', 'slices_shown', 'mean_dice']].to_string(index=False))
    print("\n✓ Done.")

if __name__ == "__main__":
    main()