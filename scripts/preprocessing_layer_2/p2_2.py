import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# CONFIGURATION
# ============================================================

SET_DIR     = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\img"
ANO_DIR     = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\ano"
PREVIEW_DIR = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\preview"

# How many images to preview
N_IMAGES = 10

# Class names (index matches OUTPUT_CLASS_ID in p2_1.py)
CLASS_NAMES = {0: "crater"}
BOX_COLOR   = (0, 255, 80)   # Bright green in BGR
BOX_THICKNESS = 2
FONT_SCALE    = 0.5


def draw_boxes_on_image(img_bgr: np.ndarray, label_path: str) -> tuple:
    """
    Reads a YOLO .txt label file and draws bounding boxes on the image.
    Returns (annotated_image_rgb, crater_count).
    """
    h, w = img_bgr.shape[:2]
    crater_count = 0

    if not os.path.isfile(label_path):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert from YOLO normalized to pixel coordinates
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        label = CLASS_NAMES.get(class_id, f"class_{class_id}")

        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Draw label background + text
        text = f"{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(img_bgr, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

        crater_count += 1

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, crater_count


def visualize_first_n(n: int = 10):
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    # Get first N images from the set directory
    supported = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    all_images = sorted([f for f in os.listdir(SET_DIR) if f.lower().endswith(supported)])

    if not all_images:
        print(f"ERROR: No images found in:\n  {SET_DIR}")
        return

    selected = all_images[:n]
    print(f"Visualizing bounding boxes for first {len(selected)} images...\n")

    # ---- Draw boxes and collect for grid ----
    annotated_images = []
    for filename in selected:
        img_path   = os.path.join(SET_DIR, filename)
        label_path = os.path.join(ANO_DIR, os.path.splitext(filename)[0] + ".txt")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [SKIP] Could not read: {filename}")
            continue

        img_rgb, count = draw_boxes_on_image(img_bgr.copy(), label_path)

        # Save individual preview
        save_path = os.path.join(PREVIEW_DIR, f"preview_{filename}")
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        annotated_images.append((img_rgb, filename, count))
        print(f"  [OK] {filename} — {count} crater(s)")

    # ---- Plot a 2x5 grid ----
    cols = 5
    rows = (len(annotated_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("OHRC Crater Annotation Preview — First 10 Images",
                 fontsize=14, color='white', fontweight='bold', y=1.01)

    axes = np.array(axes).flatten()

    for idx, (img_rgb, filename, count) in enumerate(annotated_images):
        ax = axes[idx]
        ax.imshow(img_rgb)
        ax.set_title(f"{filename}\n{count} crater(s)", fontsize=7,
                     color='#00ff50', pad=3)
        ax.axis('off')
        # Draw a subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#00ff50')
            spine.set_linewidth(1.5)

    # Hide any unused subplots
    for idx in range(len(annotated_images), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save the grid
    grid_path = os.path.join(PREVIEW_DIR, "bbox_grid_preview.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()

    print(f"\n{'='*55}")
    print(f"  Preview grid saved to:")
    print(f"  {grid_path}")
    print(f"  Individual images saved to: {PREVIEW_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    visualize_first_n(N_IMAGES)
