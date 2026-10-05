import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
import gc

try:
    import cupy as cp
    # ✅ Limit CuPy pool to 6.5 GB — leaves 1.5 GB headroom on 8 GB VRAM
    cp.get_default_memory_pool().set_limit(size=6_500 * 1024 * 1024)
    CUDA_AVAILABLE = True
    print("✅ CUDA ready — pool capped at 6.5 GB VRAM")
except ImportError:
    cp = np
    CUDA_AVAILABLE = False
    print("⚠️  CuPy not found — CPU mode")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_DIR = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\ohrc_images")
OUT_DIR = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\512_size")

TILE_SIZE        = 512
DTYPE            = np.uint8
APPLY_DESTRIPING = True

# ── Hardware-tuned constants ───────────────────────────────────────────────────
# 4060 8GB + 16GB RAM:
#   - Each chunk = GPU_CHUNK_ROWS × image_width × 4 bytes (float32)
#   - Typical OHRC width ~5000px → 4096 rows × 5000 × 4 = ~82 MB per chunk ✅
#   - We keep 2 chunks in flight (current + prefetch) = ~164 MB VRAM — very safe
#   - PNG save is I/O bound → use thread pool to overlap saves with GPU work
GPU_CHUNK_ROWS  = 4096      # rows per GPU batch (~80–100 MB per chunk)
SAVE_THREADS    = 8         # parallel PNG writers (I/O bound, not CPU bound)
# ─────────────────────────────────────────────

# ── Global thread pool for PNG saves (shared across images) ───────────────────
_save_pool = ThreadPoolExecutor(max_workers=SAVE_THREADS)
# ── GPU lock: only one chunk on GPU at a time ─────────────────────────────────
_gpu_lock  = threading.Lock()


def get_image_shape(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines, samples = None, None
        for axis_array in root.iter('{http://pds.nasa.gov/pds4/pds/v1}Axis_Array'):
            axis_name = axis_array.find('{http://pds.nasa.gov/pds4/pds/v1}axis_name')
            elements  = axis_array.find('{http://pds.nasa.gov/pds4/pds/v1}elements')
            if axis_name is not None and elements is not None:
                if axis_name.text == "Line":
                    lines = int(elements.text)
                elif axis_name.text == "Sample":
                    samples = int(elements.text)
        return lines, samples
    except Exception as e:
        print(f"❌ XML error: {xml_path.name} → {e}")
        return None, None


def compute_correction_params(raw):
    """
    Compute destripe + norm params from a small sample — never loads
    full image into RAM. Runs once per image on CPU before GPU loop.
    """
    # Sample every 16th row — for a 50k-row image that's ~3000 rows, plenty
    sampled      = raw[::16, :]
    col_median   = np.median(sampled, axis=0).astype(np.float32)   # (W,)
    global_med   = float(np.median(col_median))

    # Percentile from 1-in-400 pixels — statistically robust
    flat = raw[::20, ::20].ravel()
    p2   = float(np.percentile(flat, 2))
    p98  = float(np.percentile(flat, 98))

    return col_median, global_med, p2, p98


def process_chunk_gpu(chunk_np, col_median_gpu, global_median, p2, p98):
    """
    Process one strip entirely on GPU.
    Input  : CPU uint8 (rows, W)
    Output : CPU uint8 (rows, W)
    Peak VRAM: 2 × chunk_size (float32 + uint8)
    """
    with _gpu_lock:
        # Upload — synchronous fallback is fine at this chunk size (<100 MB)
        gpu = cp.asarray(chunk_np, dtype=cp.float32)

        if APPLY_DESTRIPING:
            gpu -= col_median_gpu    # broadcast (W,) — in-place, no alloc
            gpu += global_median

        if p98 > p2:
            gpu -= p2
            gpu /= (p98 - p2)
            cp.clip(gpu, 0.0, 1.0, out=gpu)
            gpu *= 255.0

        out = cp.asnumpy(gpu.astype(cp.uint8))   # download once
        del gpu
        # ✅ Don't free_all_blocks — pool reuse is faster
        return out


def process_chunk_cpu(chunk_np, col_median, global_median, p2, p98):
    chunk = chunk_np.astype(np.float32)
    if APPLY_DESTRIPING:
        chunk -= col_median
        chunk += global_median
    if p98 > p2:
        chunk -= p2
        chunk /= (p98 - p2)
        np.clip(chunk, 0.0, 1.0, out=chunk)
        chunk *= 255.0
    return chunk.astype(np.uint8)


def save_tile(tile, filepath):
    """Runs in thread pool — overlaps with GPU work."""
    Image.fromarray(tile).save(filepath, compress_level=0)


def process_and_stream(raw, height, width, img_stem,
                       col_median, global_med, p2, p98,
                       col_median_gpu, tile_bar, total_tiles):
    """
    Core pipeline:
      For each chunk:
        1. Read strip from memmap (CPU, fast)
        2. Send to GPU (async upload)
        3. Process on GPU
        4. Dispatch tile saves to thread pool (non-blocking)
        5. Move to next chunk while tiles are being written
    """
    count        = 0
    save_futures = []

    for y_start in range(0, height - TILE_SIZE + 1, GPU_CHUNK_ROWS):
        y_end    = min(y_start + GPU_CHUNK_ROWS, height)
        # ✅ np.array() makes a contiguous copy from memmap — safe for GPU
        chunk_np = np.array(raw[y_start:y_end, :])

        # ── GPU or CPU process ────────────────────────────────────────────
        if CUDA_AVAILABLE:
            strip_u8 = process_chunk_gpu(
                chunk_np, col_median_gpu, global_med, p2, p98
            )
        else:
            strip_u8 = process_chunk_cpu(
                chunk_np, col_median, global_med, p2, p98
            )
        del chunk_np

        strip_h = strip_u8.shape[0]

        # ── Dispatch tile saves to thread pool ────────────────────────────
        for y_local in range(0, strip_h - TILE_SIZE + 1, TILE_SIZE):
            y_abs = y_start + y_local
            for x in range(0, width - TILE_SIZE + 1, TILE_SIZE):
                tile     = strip_u8[y_local:y_local + TILE_SIZE,
                                    x:x + TILE_SIZE].copy()   # copy before del
                filepath = OUT_DIR / f"{img_stem}_y{y_abs}_x{x}.png"
                # ✅ Non-blocking — GPU starts next chunk immediately
                fut = _save_pool.submit(save_tile, tile, filepath)
                save_futures.append(fut)
                count += 1

        del strip_u8

        # ── Update tile bar ───────────────────────────────────────────────
        pct = int((count / total_tiles) * 100) if total_tiles else 100
        tile_bar.n = min(pct, 99)   # hold at 99 until saves confirmed
        tile_bar.set_postfix_str(f"{count}/{total_tiles} tiles")
        tile_bar.refresh()

    # ── Wait for all saves to finish before returning ─────────────────────
    for fut in save_futures:
        fut.result()

    return count


def process_image(args):
    img_path, tile_bar = args

    xml_path = img_path.with_suffix(".xml")
    if not xml_path.exists():
        return 0, img_path.name, "no xml"

    height, width = get_image_shape(xml_path)
    if height is None or width is None:
        return 0, img_path.name, "bad xml"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_tiles = (height // TILE_SIZE) * (width // TILE_SIZE)

    try:
        # ✅ memmap: no RAM used until we slice it
        raw = np.memmap(img_path, dtype=DTYPE, mode='r', shape=(height, width))

        # ── Pre-compute params from samples (CPU, ~50 ms) ─────────────────
        col_median, global_med, p2, p98 = compute_correction_params(raw)

        # ── Upload col_median to GPU once (W floats — tiny) ───────────────
        col_median_gpu = cp.asarray(col_median) if CUDA_AVAILABLE else col_median

        tile_bar.set_description(f"🧩 {img_path.name[:22]:<22}")
        tile_bar.n = 0
        tile_bar.refresh()

        count = process_and_stream(
            raw, height, width, img_path.stem,
            col_median, global_med, p2, p98,
            col_median_gpu, tile_bar, total_tiles
        )

        # ── Cleanup ───────────────────────────────────────────────────────
        del raw, col_median, col_median_gpu
        if CUDA_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    except Exception as e:
        return 0, img_path.name, f"error: {e}"

    tile_bar.n = 100
    tile_bar.set_postfix_str(f"{count}/{total_tiles} ✅")
    tile_bar.refresh()

    return count, img_path.name, "ok"


def main():
    img_files = list(IMG_DIR.glob("*.img"))
    total     = len(img_files)
    print(f"🔍 Found {total} images")
    print(f"⚡ GPU chunk: {GPU_CHUNK_ROWS} rows | Save threads: {SAVE_THREADS}\n")

    img_bar  = tqdm(total=total, desc="📦 Overall", unit="img",
                    position=0, leave=True)
    tile_bar = tqdm(total=100,   desc="🧩 Waiting ",
                    position=1,  leave=False,
                    bar_format="{desc} {bar} {n:3d}% | {postfix}")

    total_tiles    = 0
    completed_imgs = 0

    # ✅ Single process — GPU is shared, multiprocessing would fight over VRAM
    # Thread pool inside handles I/O parallelism instead
    for img_path in img_files:
        tiles, name, status = process_image((img_path, tile_bar))

        completed_imgs += 1
        total_tiles    += tiles
        img_bar.update(1)
        img_bar.set_postfix({
            "done"  : f"{completed_imgs}/{total}",
            "tiles" : total_tiles,
            "status": status
        })

    img_bar.close()
    tile_bar.close()
    _save_pool.shutdown(wait=True)

    print(f"\n✅ DONE — {completed_imgs}/{total} images | {total_tiles} tiles total")


if __name__ == "__main__":
    main()