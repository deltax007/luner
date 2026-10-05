"""
annotate_all.py  —  OPTIMISED
-------------------------------
YOLOv5 annotation pipeline with:
  • FP16 (half-precision)  — ~2× faster inference, ~50% less VRAM
  • Threaded prefetch      — next batch loaded from disk while GPU runs
  • Async file I/O         — annotation writes / image copies don't block GPU
  • OOM auto-recovery      — halves batch size on CUDA OOM and retries
  • 8 GB hard cap          — stops when IMG_DIR reaches limit
  • Resume support         — skips already-annotated tiles
"""

import cv2
import gc
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

# ═══════════════════════════  CONFIG  ════════════════════════════════════════

MODEL_PATH       = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\scripts\testing\best.pt"
IMAGE_DIR        = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\512_size"
ANO_DIR          = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\ano"
IMG_DIR          = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\img"

CONF_THRESH      = 0.70   # Detection confidence threshold
BATCH_SIZE       = 64     # Images per GPU forward pass (auto-halves on OOM)
SIZE_LIMIT_GB    = 8.0    # Hard stop when IMG_DIR reaches this

PREFETCH_WORKERS = 8      # Threads for reading images from disk
IO_WORKERS       = 6      # Threads for writing .txt + copying images
USE_HALF         = True   # FP16 inference (requires CUDA)
INFER_SIZE       = 512    # Model input size (matches tile size → no resize)
CACHE_FLUSH_N    = 30     # Flush GPU cache every N batches

# ═════════════════════════════════════════════════════════════════════════════

SIZE_LIMIT_BYTES = int(SIZE_LIMIT_GB * 1024 ** 3)
TILE_W = TILE_H  = 512


# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    print("[INFO] Loading YOLOv5 model …")
    model = torch.hub.load(
        "ultralytics/yolov5", "custom",
        path=model_path,
        force_reload=False,
        verbose=False,
    )
    model.conf = CONF_THRESH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if USE_HALF and device == "cuda":
        model.half()

    if device == "cuda":
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        fp   = "FP16" if USE_HALF else "FP32"
        print(f"[INFO] GPU    : {gpu}  ({vram:.1f} GB VRAM)  [{fp}]")
    else:
        print("[WARN] CUDA not available — running on CPU (very slow)")

    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Conf   : {CONF_THRESH}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
def to_yolo(x1, y1, x2, y2, W=TILE_W, H=TILE_H):
    """Pixel xyxy → YOLO normalised cx cy w h, clamped to [0, 1]."""
    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / W))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / H))
    w  = max(0.0, min(1.0, (x2 - x1) / W))
    h  = max(0.0, min(1.0, (y2 - y1) / H))
    return cx, cy, w, h


def read_image(path: Path):
    """Thread worker: read one image via cv2. Returns (path, array|None)."""
    return path, cv2.imread(str(path))


def write_result(path: Path, det_df, ano_dir: Path, img_dir: Path) -> int:
    """Thread worker: write YOLO .txt and copy image. Returns file size."""
    lines = []
    for _, row in det_df.iterrows():
        cls_id = int(row["class"])
        cx, cy, w, h = to_yolo(row.xmin, row.ymin, row.xmax, row.ymax)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (ano_dir / (path.stem + ".txt")).write_text("\n".join(lines))
    shutil.copy2(path, img_dir / path.name)
    return path.stat().st_size


def dir_size_bytes(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    except StopIteration:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ano_dir = Path(ANO_DIR)
    img_dir = Path(IMG_DIR)
    ano_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-annotated stems
    already_done = {p.stem for p in ano_dir.glob("*.txt")}
    if already_done:
        print(f"[INFO] Resume: {len(already_done):,} tiles already annotated — skipping.")

    model, device = load_model(MODEL_PATH)

    all_imgs = sorted(
        list(Path(IMAGE_DIR).glob("*.png")) +
        list(Path(IMAGE_DIR).glob("*.jpg"))
    )
    todo = [p for p in all_imgs if p.stem not in already_done]

    copied_bytes = dir_size_bytes(img_dir)

    print(f"\n[INFO] Total tiles : {len(all_imgs):,}")
    print(f"[INFO] To process  : {len(todo):,}")
    print(f"[INFO] Size limit  : {SIZE_LIMIT_GB} GB")
    if copied_bytes:
        print(f"[INFO] Already in IMG_DIR: {copied_bytes / 1024**3:.2f} GB")
    print()

    total_saved   = 0
    total_skipped = 0
    limit_reached = False
    current_bs    = BATCH_SIZE

    pbar = tqdm(total=len(todo), desc="Annotating", unit="img", dynamic_ncols=True)
    pbar.set_postfix(saved=0, skip=0, gb=f"0.00/{SIZE_LIMIT_GB}", bs=current_bs)

    # Separate pools: one for prefetch (I/O bound), one for saving results
    prefetch_pool = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)
    io_pool       = ThreadPoolExecutor(max_workers=IO_WORKERS)

    i = 0
    batch_count = 0

    try:
        while i < len(todo) and not limit_reached:

            batch_paths = todo[i : i + current_bs]
            i += current_bs

            # ── Parallel image read ───────────────────────────────────────────
            read_futures = {prefetch_pool.submit(read_image, p): p for p in batch_paths}
            imgs, valid_paths = [], []
            for fut in as_completed(read_futures):
                path, img = fut.result()
                if img is None:
                    total_skipped += 1
                    pbar.update(1)
                    continue
                imgs.append(img)
                valid_paths.append(path)

            if not imgs:
                continue

            # ── GPU inference (FP16 if enabled) ──────────────────────────────
            try:
                with torch.no_grad():
                    results = model(imgs, size=INFER_SIZE)
                det_dfs = results.pandas().xyxy

            except torch.cuda.OutOfMemoryError:
                # Auto-recover: halve batch size and retry this batch
                current_bs = max(8, current_bs // 2)
                print(f"\n[WARN] CUDA OOM — reducing batch to {current_bs}, retrying …")
                torch.cuda.empty_cache()
                gc.collect()
                i -= len(imgs)          # retry the same images
                del imgs
                continue

            # ── Submit async I/O per detected image ───────────────────────────
            io_futures = []
            for path, det_df in zip(valid_paths, det_dfs):
                if len(det_df) == 0:
                    total_skipped += 1
                    pbar.update(1)
                    continue

                fut = io_pool.submit(write_result, path, det_df, ano_dir, img_dir)
                io_futures.append((fut, path.stat().st_size))
                total_saved  += 1
                copied_bytes += path.stat().st_size
                pbar.update(1)
                pbar.set_postfix({
                    "saved": f"{total_saved:,}",
                    "skip":  f"{total_skipped:,}",
                    "gb":    f"{copied_bytes / 1024**3:.2f}/{SIZE_LIMIT_GB}",
                    "bs":    current_bs,
                })

                if copied_bytes >= SIZE_LIMIT_BYTES:
                    limit_reached = True
                    break

            # ── Memory cleanup ────────────────────────────────────────────────
            del results, det_dfs, imgs
            batch_count += 1
            if batch_count % CACHE_FLUSH_N == 0 and device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    finally:
        pbar.close()

        # Drain remaining I/O futures
        print("[INFO] Flushing file I/O …", end=" ", flush=True)
        prefetch_pool.shutdown(wait=False)
        io_pool.shutdown(wait=True)
        print("done.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    tag = "✓  8 GB limit reached." if limit_reached else "✓  All tiles processed."
    print(f"[DONE] {tag}")
    print(f"       Images saved   : {total_saved:,}")
    print(f"       Images skipped : {total_skipped:,}  (conf < {CONF_THRESH})")
    print(f"       Total copied   : {copied_bytes / 1024**3:.3f} GB")
    print(f"       Annotations    : {ANO_DIR}")
    print(f"       Images         : {IMG_DIR}")

    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
