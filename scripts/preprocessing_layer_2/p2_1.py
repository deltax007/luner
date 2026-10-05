import os
import gc
import time
import shutil
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

# Source: all 30GB of OHRC tiles
IMAGE_DIR  = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\512_size"

# Destination: only images WITH craters go here (~8GB subset)
SET_DIR    = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\set"

# Destination: YOLO .txt annotation files (paired with images in SET_DIR)
ANO_DIR    = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\ano"

MODEL_PATH = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\scripts\testing\best (1).pt"

# Confidence threshold (0.0 - 1.0)
CONF_THRESHOLD = 0.85

# -----------------------------------------------------------------------
# CLASS FILTER
# Model classes: {0: Boulder, 1: Crater, 2: Plain surface}
# We ONLY want Craters (class 1).
# In the output .txt files, Crater will be remapped to class 0
# (since it's the only class we're saving).
# -----------------------------------------------------------------------
CRATER_CLASS_ID   = 1   # Original class ID in the model
OUTPUT_CLASS_ID   = 0   # Remapped output class ID (0-indexed, single class)

# -----------------------------------------------------------------------
# BATCH SIZE
# For 150k images at ~256KB each, batching is critical for speed.
# Recommended:
#   GPU (CUDA available) : 32–64
#   CPU only             : 4–8  (higher = more RAM used)
# -----------------------------------------------------------------------
BATCH_SIZE = 32

# Free GPU cache every N batches to prevent memory creep
CLEAR_CACHE_EVERY_N_BATCHES = 10

# -----------------------------------------------------------------------
# 8 GB HARD LIMIT for final_set/set/
# The script stops copying images once the SET_DIR reaches this size.
# The .txt files in ANO_DIR are tiny (negligible), so this effectively
# caps your labelled dataset at exactly 8GB.
# -----------------------------------------------------------------------
MAX_SET_SIZE_GB    = 8
MAX_SET_SIZE_BYTES = MAX_SET_SIZE_GB * 1024 ** 3   # 8_589_934_592 bytes

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ============================================================
# GPU DETECTION
# ============================================================

def get_device() -> str:
    """
    Detects whether CUDA (NVIDIA GPU) is available.
    Returns 'cuda' if a GPU is found, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [GPU] CUDA is AVAILABLE")
        print(f"        Device : {gpu_name}")
        print(f"        VRAM   : {vram:.1f} GB")
        return "cuda"
    else:
        print("  [CPU] CUDA is NOT available — running on CPU.")
        print("        Tip: Install PyTorch with CUDA support for 10–50x speedup.")
        print("        Visit: https://pytorch.org/get-started/locally/")
        return "cpu"


# ============================================================
# MEMORY HELPERS
# ============================================================

def free_memory(device: str):
    """Frees Python garbage + GPU VRAM."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def log_gpu_memory():
    """Prints current GPU VRAM usage. Only called if CUDA is active."""
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0)  / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [VRAM] {used:.2f} GB used / {total:.1f} GB total")


# ============================================================
# BATCH PROCESSOR
# ============================================================

def process_batch(model, batch_paths: list, device: str,
                  processed_count: int, skipped_count: int,
                  total_detections: int, set_size_bytes: int):
    """   
    Runs inference on a batch of image paths.
    For each image WITH craters (and while under the 8GB limit):
      - Copies the image  →  final_set/set/
      - Writes the labels →  final_set/ano/
    Images with NO craters are ignored.
    Returns updated (processed_count, skipped_count, total_detections, set_size_bytes, limit_reached).
    limit_reached is True when the 8GB cap is hit — the caller should stop immediately.
    """
    limit_reached = False

    try:
        results = model(batch_paths, conf=CONF_THRESHOLD, device=device, verbose=False)
    except Exception as e:
        print(f"  [BATCH ERROR] Entire batch failed: {e}")
        skipped_count += len(batch_paths)
        return processed_count, skipped_count, total_detections, set_size_bytes, limit_reached

    for result, img_path in zip(results, batch_paths):
        filename = os.path.basename(img_path)

        try:
            # Filter: keep only Crater boxes (original class 1)
            boxes = [b for b in result.boxes
                     if int(b.cls[0].item()) == CRATER_CLASS_ID]

            # No craters found → skip entirely (don't copy image, don't write .txt)
            if len(boxes) == 0:
                processed_count += 1
                continue

            # --- 8GB LIMIT CHECK ---
            # Get the file size before copying; stop if it would push us over the limit
            img_size = os.path.getsize(img_path)
            if set_size_bytes + img_size > MAX_SET_SIZE_BYTES:
                # Limit reached — signal caller to stop the entire pipeline
                limit_reached = True
                break

            # 1. Copy the image tile into final_set/set/
            dest_img = os.path.join(SET_DIR, filename)
            shutil.copy2(img_path, dest_img)
            set_size_bytes += img_size  # accumulate bytes copied

            # 2. Write YOLO .txt annotation into final_set/ano/
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(ANO_DIR, txt_filename)

            with open(txt_filepath, 'w') as f:
                for box in boxes:
                    # Normalized coordinates (0–1), required by YOLO format
                    x_c, y_c, w, h = box.xywhn[0].tolist()
                    # Remap Crater (class 1) → class 0 (single-class output)
                    f.write(f"{OUTPUT_CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            total_detections += len(boxes)
            processed_count  += 1

        except Exception as e:
            print(f"  [SKIP] {filename} — post-processing error: {e}")
            skipped_count += 1

    # Free result tensors from GPU immediately after each batch
    del results
    free_memory(device)

    return processed_count, skipped_count, total_detections, set_size_bytes, limit_reached


# ============================================================
# MAIN PIPELINE
# ============================================================

def auto_annotate():
    print("=" * 60)
    print("  OHRC CRATER AUTO-ANNOTATION  (YOLOv8)")
    print("=" * 60)

    # --- Validate paths ---
    if not os.path.isdir(IMAGE_DIR):
        print(f"\nERROR: Image directory not found:\n  {IMAGE_DIR}")
        return
    if not os.path.isfile(MODEL_PATH):
        print(f"\nERROR: Model weights not found:\n  {MODEL_PATH}")
        return

    # Ensure both output folders exist
    os.makedirs(SET_DIR, exist_ok=True)
    os.makedirs(ANO_DIR, exist_ok=True)

    # --- GPU Detection ---
    print("\n[1/4] Checking hardware ...")
    device = get_device()

    # --- Load model ---
    print("\n[2/4] Loading YOLOv8 model ...")
    try:
        model = YOLO(MODEL_PATH)
        # Move model to GPU once — reused for all batches
        model.to(device)
    except Exception as e:
        print(f"ERROR: Failed to load model — {e}")
        return
    print(f"  Model loaded successfully.")
    print(f"  Classes : {model.names}")
    print(f"  Keeping : class {CRATER_CLASS_ID} (Crater) → remapped to class {OUTPUT_CLASS_ID}")

    # --- Collect images ---
    print("\n[3/4] Scanning image directory ...")
    image_files = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    total_images = len(image_files)

    if total_images == 0:
        print(f"  WARNING: No supported images found in:\n  {IMAGE_DIR}")
        return
    print(f"  Found {total_images:,} images to process.")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Device     : {device.upper()}")

    # --- Batch processing ---
    print(f"\n[4/4] Running inference ...\n")
    processed_count  = 0
    skipped_count    = 0
    total_detections = 0
    set_size_bytes   = 0   # tracks total bytes copied into SET_DIR
    batch_num        = 0
    start_time       = time.time()

    # tqdm bar — tracks individual images, updates after every batch
    pbar = tqdm(
        total=total_images,
        unit="img",
        desc="Annotating",
        colour="cyan",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    # Split 150k images into chunks of BATCH_SIZE
    limit_reached = False
    for i in range(0, total_images, BATCH_SIZE):
        batch_paths = image_files[i : i + BATCH_SIZE]
        batch_num  += 1

        processed_count, skipped_count, total_detections, set_size_bytes, limit_reached = process_batch(
            model, batch_paths, device,
            processed_count, skipped_count, total_detections, set_size_bytes
        )

        # Advance bar by however many images were in this batch
        pbar.update(len(batch_paths))

        # Update the live stats shown on the right of the bar
        pbar.set_postfix({
            "craters" : total_detections,
            "set_GB"  : f"{set_size_bytes / 1024**3:.2f}/{MAX_SET_SIZE_GB}GB",
            "skipped" : skipped_count,
            "device"  : device.upper()
        })

        # --- 8GB LIMIT CHECK — stop immediately if signalled by process_batch ---
        if limit_reached:
            pbar.set_description("Annotating [8GB LIMIT REACHED — STOPPED]")
            print(f"\n  [STOP] 8 GB limit reached after batch {batch_num}. Pipeline halted.")
            break

        # --- Periodic memory cleanup ---
        if batch_num % CLEAR_CACHE_EVERY_N_BATCHES == 0:
            free_memory(device)
            if device == "cuda":
                log_gpu_memory()

    pbar.close()

    # --- Final summary ---
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  AUTO-ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"  Total images     : {total_images:,}")
    print(f"  Processed        : {processed_count:,}")
    print(f"  Skipped (errors) : {skipped_count:,}")
    print(f"  Total craters    : {total_detections:,}")
    print(f"  Dataset size     : {set_size_bytes / 1024**3:.2f} GB / {MAX_SET_SIZE_GB} GB limit")
    print(f"  Limit reached    : {'YES — stopped early' if limit_reached else 'No — all images processed'}")
    print(f"  Time taken       : {total_time/60:.1f} minutes")
    print(f"  Avg speed        : {total_images/total_time:.1f} img/s")
    print(f"  Tiles saved to   : {SET_DIR}")
    print(f"  Labels saved to  : {ANO_DIR}")
    print("=" * 60)

    # Write classes.txt into the ano folder so CVAT/LabelImg reads it correctly
    classes_file = os.path.join(ANO_DIR, "classes.txt")
    with open(classes_file, 'w') as f:
        f.write("crater\n")
    print(f"\n  classes.txt written to: {classes_file}")


if __name__ == "__main__":
    auto_annotate()
