"""
predict_preview.py
------------------
Runs best.pt on 5 randomly sampled tile images and displays a
matplotlib grid preview with YOLO bounding-box overlays.

Usage:
    python predict_preview.py

Optional overrides (edit the CONFIG section below):
    MODEL_PATH  – path to the .pt weights file
    IMAGE_DIR   – directory to sample images from
    N_IMAGES    – how many images to preview (default 5)
    CONF_THRESH – confidence threshold for detections
"""

import os
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# ─────────────────────────────  CONFIG  ─────────────────────────────────────
MODELS      = {
    "Best": r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\models\yolo26n_craters_v1-5\weights\best.pt",
    "Last": r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\models\yolo26n_craters_v1-5\weights\last.pt"
}
IMAGE_DIR   = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\img"
N_IMAGES    = 5
CONF_THRESH = 0.45
SEED        = 42          # set to None for truly random picks every run
# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_path: str):
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Classes: {model.names}")
    return model


def sample_images(image_dir: str, n: int, seed) -> list[Path]:
    """Pick `n` random PNG/JPG images from image_dir."""
    img_dir = Path(image_dir)
    all_imgs = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    if not all_imgs:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    if seed is not None:
        random.seed(seed)
    picks = random.sample(all_imgs, min(n, len(all_imgs)))
    print(f"[INFO] Sampled {len(picks)} images from {img_dir}")
    return picks




def draw_boxes(img_bgr: np.ndarray, det_df, class_names: dict) -> np.ndarray:
    """
    Draw bounding boxes on a copy of the image.
    det_df : pandas DataFrame with columns xmin, ymin, xmax, ymax, confidence, class, name
    Returns RGB numpy array.
    """
    img = img_bgr.copy()
    palette = [
        (0, 200, 255), (255, 100, 0), (50, 220, 50),
        (200, 0, 200), (255, 220, 0), (0, 160, 255),
    ]

    for _, row in det_df.iterrows():
        x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
        cls_id = int(row["class"])
        conf   = float(row["confidence"])
        label  = row["name"]
        color  = palette[cls_id % len(palette)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_preview(image_paths, models_det_dfs, images_bgr, class_names):
    """Render a dark-themed matplotlib grid of annotated images comparing models."""
    n_models = len(models_det_dfs)
    n_imgs   = len(image_paths)
    fig = plt.figure(figsize=(5 * n_imgs, 5.5 * n_models), facecolor="#0d0d0d")
    fig.suptitle(
        "YOLO26 Crater Detection — Model Comparison",
        fontsize=16, fontweight="bold", color="white", y=1.02,
    )

    for row_idx, (model_name, det_dfs) in enumerate(models_det_dfs.items()):
        for col_idx, (path, det_df, img_bgr) in enumerate(zip(image_paths, det_dfs, images_bgr)):
            annotated = draw_boxes(img_bgr, det_df, class_names)
            n_det     = len(det_df)

            plot_idx = row_idx * n_imgs + col_idx + 1
            ax = fig.add_subplot(n_models, n_imgs, plot_idx)
            ax.imshow(annotated)
            
            title = f"[{model_name}] {path.name}\n{n_det} detection{'s' if n_det != 1 else ''}"
            ax.set_title(title, fontsize=8, color="#e0e0e0", pad=4)
            ax.axis("off")

    plt.tight_layout(pad=0.8)

    # Save to the same folder as the script and auto-open it
    out_path = Path(__file__).parent / "preview_output.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[INFO] Preview saved → {out_path}")
    os.startfile(out_path)   # opens with default image viewer on Windows


def main():
    image_paths = sample_images(IMAGE_DIR, N_IMAGES, SEED)
    images_bgr  = [cv2.imread(str(p)) for p in image_paths]

    models_det_dfs = {}
    class_names = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}\n[INFO] Evaluating {model_name} Model\n{'='*70}")
        model = load_model(model_path)
        class_names = model.names

        print(f"[INFO] Running inference …")
        results = model([str(p) for p in image_paths], conf=CONF_THRESH)

        det_dfs = []
        for r in results:
            boxes = r.boxes.data.cpu().numpy() if r.boxes is not None else np.empty((0, 6))
            df = pd.DataFrame(boxes, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"])
            df["name"] = df["class"].apply(lambda x: model.names[int(x)] if not pd.isna(x) else "")
            det_dfs.append(df)
            
        models_det_dfs[model_name] = det_dfs

        # ── Summary table ────────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print(f"{'Image':<45} {'Detections':>10}")
        print("=" * 65)
        for path, det_df in zip(image_paths, det_dfs):
            print(f"{path.name:<45} {len(det_df):>10}")
            for _, row in det_df.iterrows():
                xyxy = [round(v, 1) for v in [row.xmin, row.ymin, row.xmax, row.ymax]]
                print(f"    {row['name']:>15}  conf={row['confidence']:.3f}  box={xyxy}")
        print("=" * 65)

    show_preview(image_paths, models_det_dfs, images_bgr, class_names)


if __name__ == "__main__":
    main()
