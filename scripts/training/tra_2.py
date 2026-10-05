"""
tra_2.py  —  YOLO26n Crater Detection Training
-------------------------------------------------
Trains YOLO26n on the annotated lunar crater dataset.

Pipeline:
  1. Reads images from IMG_DIR and labels from LABEL_DIR
  2. Splits into train / val sets (85 / 15 %)
  3. Builds the required ultralytics directory layout using hardlinks
     (no extra disk space consumed)
  4. Writes data.yaml
  5. Trains YOLO26n with CUDA + recommended hyperparameters

Usage:
    & "C:/Users/ayush/anaconda3/envs/flare/python.exe" `
      "c:/Users/ayush/OneDrive/Documents/prada_pro/coc/scripts/training/tra_2.py"
"""

import gc
import os
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from tqdm import tqdm

# ═══════════════════════════  PATHS  ═════════════════════════════════════════

IMG_DIR     = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\img"
LABEL_DIR   = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\ano"
DATASET_DIR = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\yolo_dataset"
MODEL_DIR   = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\models"
RUN_NAME    = "yolo26n_craters_v1"

# ═══════════════════════  TRAINING CONFIG  ════════════════════════════════════

BASE_MODEL   = r"C:\Users\ayush\OneDrive\Documents\prada_pro\yolo26n.pt"
EPOCHS       = 150            # max epochs (early stopping will cut it short)
BATCH_SIZE   = 32             # increased to 32 to maximize 8GB VRAM usage
IMG_SIZE     = 512
VAL_SPLIT    = 0.15           # 15 % of data for validation
PATIENCE     = 30             # early stopping patience (epochs without improvement)
WORKERS      = 4              # 4 workers provides a good speedup without overloading Windows shared memory
SEED         = 42
RESUME       = False          # set True to resume an interrupted run

# ── Optimizer & LR ──────────────────────────────────────────────────────────
OPTIMIZER    = "MuSGD"        # YOLO26 recommended optimizer
LR0          = 0.01           # initial learning rate
LRF          = 0.01           # final LR = LR0 * LRF
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3

# ── Augmentation ─────────────────────────────────────────────────────────────
MOSAIC       = 1.0    # mosaic (critical for crater scale variety)
MIXUP        = 0.15   # mixup alpha
COPY_PASTE   = 0.3    # copy-paste (helps with sparse craters)
FLIPUD       = 0.5    # vertical flip
FLIPLR       = 0.5    # horizontal flip
DEGREES      = 45.0   # rotation range (craters are rotationally symmetric)
SCALE        = 0.5    # scale jitter
HSV_H        = 0.015  # hue jitter (grayscale OHRC → keep low)
HSV_S        = 0.4    # saturation jitter
HSV_V        = 0.4    # value/brightness jitter

# ── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Crater"]
NC           = len(CLASS_NAMES)

# ── Regularisation (important: labels are pseudo-labels from YOLOv5) ─────────
LABEL_SMOOTHING = 0.1   # prevents overconfidence on noisy auto-annotations
DROPOUT         = 0.05  # mild dropout for regularisation

# ── Training tricks ───────────────────────────────────────────────────────────
MULTI_SCALE  = False    # DISABLED — randomly scales to 768px → OOM on 8GB
                        # (re-enable only if you drop batch further or upgrade GPU)
NBS          = 64       # nominal batch size; accumulates gradients over
                        # NBS/BATCH_SIZE = 2 steps → effective batch of 64

# ═════════════════════════════════════════════════════════════════════════════


def check_gpu():
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — training on CPU will be very slow!")
        return "cpu"
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[INFO] GPU  : {gpu}  ({vram:.1f} GB VRAM)")
    print(f"[INFO] CUDA : {torch.version.cuda}")
    return 0   # device index for ultralytics


def get_paired_files(img_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs where both files exist."""
    pairs = []
    img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    for img_path in img_files:
        lbl_path = label_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def link_or_copy(src: Path, dst: Path):
    """Try hardlink first (zero extra disk space), fall back to copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def setup_dataset(pairs: list, dataset_dir: Path, val_split: float, seed: int) -> Path:
    """
    Build ultralytics-compatible directory layout:
        dataset_dir/
            images/train/   images/val/
            labels/train/   labels/val/
    Uses hardlinks so no extra disk space is used.
    """
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n_val   = max(1, int(len(shuffled) * val_split))
    val_set = shuffled[:n_val]
    trn_set = shuffled[n_val:]

    splits = {"train": trn_set, "val": val_set}

    for split, items in splits.items():
        img_out = dataset_dir / "images" / split
        lbl_out = dataset_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Linking {split:5s}: {len(items):>6,} images …")
        for img_path, lbl_path in tqdm(items, desc=f"  {split}", unit="file",
                                        dynamic_ncols=True, leave=False):
            link_or_copy(img_path, img_out / img_path.name)
            link_or_copy(lbl_path, lbl_out / lbl_path.name)

    print(f"[INFO] Dataset ready: {len(trn_set):,} train  /  {len(val_set):,} val")
    return dataset_dir


def write_yaml(dataset_dir: Path) -> Path:
    """Write the ultralytics data.yaml."""
    yaml_content = f"""# YOLO26n Crater Detection — Data Config
path: {dataset_dir.as_posix()}
train: images/train
val:   images/val

nc: {NC}
names: {CLASS_NAMES}
"""
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] data.yaml → {yaml_path}")
    return yaml_path


def train(yaml_path: Path, device):
    """Load YOLO26n and run training with recommended settings."""
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print("[INFO] Starting training …\n")
    print("=" * 70)

    results = model.train(
        # ── Dataset ───────────────────────────────────────────────────────
        data       = str(yaml_path),
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,

        # ── Training schedule ─────────────────────────────────────────────
        epochs     = EPOCHS,
        patience   = PATIENCE,          # early stopping
        warmup_epochs = WARMUP_EPOCHS,

        # ── Hardware ──────────────────────────────────────────────────────
        device      = device,
        workers     = WORKERS,
        amp         = False,             # YOLO26 has FP32-only layers
        cache       = "disk",            # cache preprocessed imgs to disk

        # ── Optimizer ─────────────────────────────────────────────────────
        optimizer     = OPTIMIZER,
        lr0           = LR0,
        lrf           = LRF,
        momentum      = MOMENTUM,
        weight_decay  = WEIGHT_DECAY,
        cos_lr        = True,           # cosine LR annealing

        # ── Augmentation ──────────────────────────────────────────────────
        mosaic      = MOSAIC,
        mixup       = MIXUP,
        copy_paste  = COPY_PASTE,
        flipud      = FLIPUD,
        fliplr      = FLIPLR,
        degrees     = DEGREES,
        scale       = SCALE,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,
        multi_scale = MULTI_SCALE,       # random resize each batch

        # ── Regularisation ───────────────────────────────────────────
        dropout         = DROPOUT,
        nbs             = NBS,           # gradient accumulation target

        # ── Output ────────────────────────────────────────────────────────
        project    = str(model_dir),
        name       = RUN_NAME,
        exist_ok   = RESUME,
        resume     = RESUME,
        save       = True,              # save best.pt and last.pt
        save_period = 10,               # checkpoint every 10 epochs
        plots      = True,              # save training plots

        # ── Misc ──────────────────────────────────────────────────────────
        seed       = SEED,
        verbose    = True,
        rect       = False,             # disable rectangular batches (use mosaic)
        close_mosaic = 10,              # disable mosaic in last 10 epochs
    )

    return results


def print_summary(results, model_dir: Path):
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    run_path = model_dir / RUN_NAME
    best_pt  = run_path / "weights" / "best.pt"
    last_pt  = run_path / "weights" / "last.pt"

    if best_pt.exists():
        print(f"  Best weights : {best_pt}")
    if last_pt.exists():
        print(f"  Last weights : {last_pt}")

    # Validate best model
    if best_pt.exists():
        print("\n[INFO] Validating best.pt …")
        val_model   = YOLO(str(best_pt))
        yaml_path   = Path(DATASET_DIR) / "data.yaml"
        val_results = val_model.val(data=str(yaml_path), imgsz=IMG_SIZE,
                                    device=0 if torch.cuda.is_available() else "cpu",
                                    verbose=False)
        box = val_results.box
        print(f"  mAP@0.5      : {box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {box.map:.4f}")
        print(f"  Precision    : {box.mp:.4f}")
        print(f"  Recall       : {box.mr:.4f}")

    print("=" * 70)

    # Final GPU flush
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def clear_disk_cache(dataset_dir: Path):
    """Delete all .cache files ultralytics wrote during training."""
    cache_files = list(dataset_dir.rglob("*.cache"))
    if not cache_files:
        print("[INFO] No disk cache files found — nothing to clean.")
        return
    total_mb = sum(f.stat().st_size for f in cache_files) / 1024 ** 2
    print(f"\n[INFO] Clearing disk cache: {len(cache_files)} file(s)  ({total_mb:.1f} MB) …")
    for f in cache_files:
        try:
            f.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete {f.name}: {e}")
    print("[INFO] Disk cache cleared.")


def main():
    print("=" * 70)
    print("  YOLO26n — Lunar Crater Training Pipeline")
    print("=" * 70)

    device = check_gpu()

    img_dir   = Path(IMG_DIR)
    label_dir = Path(LABEL_DIR)

    # ── Step 1: Pair images with annotations ─────────────────────────────
    print(f"\n[1/4] Scanning dataset …")
    pairs = get_paired_files(img_dir, label_dir)
    if not pairs:
        raise RuntimeError(f"No paired image+label files found.\n"
                           f"  IMG_DIR  : {img_dir}\n"
                           f"  LABEL_DIR: {label_dir}")
    print(f"[INFO] Found {len(pairs):,} paired image+label files")

    # ── Step 2: Build directory layout ───────────────────────────────────
    print(f"\n[2/4] Setting up dataset layout …")
    dataset_dir = Path(DATASET_DIR)
    setup_dataset(pairs, dataset_dir, VAL_SPLIT, SEED)

    # ── Step 3: Write data.yaml ───────────────────────────────────────────
    print(f"\n[3/4] Writing data.yaml …")
    yaml_path = write_yaml(dataset_dir)

    # ── Step 4: Train ─────────────────────────────────────────────────────
    print(f"\n[4/4] Training YOLO26n …")
    print(f"      Epochs       : {EPOCHS}  (early stop patience: {PATIENCE})")
    print(f"      Batch size   : {BATCH_SIZE}")
    print(f"      Image size   : {IMG_SIZE} px")
    print(f"      Optimizer    : {OPTIMIZER}")
    print(f"      Output dir   : {MODEL_DIR}/{RUN_NAME}")
    print()

    results = train(yaml_path, device)
    print_summary(results, Path(MODEL_DIR))

    # ── Step 5: Clean disk cache ──────────────────────────────────────────
    print(f"\n[5/5] Cleaning disk cache …")
    clear_disk_cache(Path(DATASET_DIR))


if __name__ == "__main__":
    main()