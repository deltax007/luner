"""
tra_1.py  —  YOLO Version Benchmark
--------------------------------------
Tests YOLOv8 / YOLOv9 / YOLOv10 / YOLOv11 at every model size
on your actual 512-px lunar tiles.

Reports per model:
  • Inference speed  (FPS & ms/img)
  • Peak VRAM usage  (MB)
  • Parameter count
  • Model file size  (MB)

Run with:
    & "C:/Users/ayush/anaconda3/envs/flare/python.exe" `
      "c:/Users/ayush/OneDrive/Documents/prada_pro/coc/scripts/training/tra_1.py"
"""

import gc
import random
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ═══════════════════════════  CONFIG  ════════════════════════════════════════

IMAGE_DIR    = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\final_set\img"
N_WARMUP     = 10      # warm-up passes before timing
N_BENCH      = 100     # images to time per model
BATCH_SIZE   = 32      # inference batch size
IMG_SIZE     = 512     # must match tile size
SEED         = 42

# Models to benchmark (auto-downloaded on first run if not cached)
# Format: (display_name, ultralytics_model_id)
MODELS = [
    # ── YOLO26 (released Jan 14 2026) ────────────────────────────────────────
    # NMS-free, ProgLoss + STAL for small objects — ideal for crater detection
    ("YOLO26n",  "yolo26n.pt"),
    ("YOLO26s",  "yolo26s.pt"),
    ("YOLO26m",  "yolo26m.pt"),
    ("YOLO26l",  "yolo26l.pt"),
    ("YOLO26x",  "yolo26x.pt"),
]

# Training time estimation ───────────────────────────────────────────────────
TRAIN_IMAGES    = 32_685   # your 8 GB dataset size
TRAIN_EPOCHS    = 100      # standard full training run
BACKPROP_FACTOR = 3.0      # training is ~3× slower than pure inference
                           # (forward + backward + optimizer + augmentation)
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF   = False   # YOLO26 has FP32-only layers — FP16 causes dtype mismatch


def sample_images(image_dir: str, n: int, seed: int) -> list:
    all_imgs = sorted(
        list(Path(image_dir).glob("*.png")) +
        list(Path(image_dir).glob("*.jpg"))
    )
    random.seed(seed)
    picks = random.sample(all_imgs, min(n, len(all_imgs)))
    imgs = []
    for p in picks:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    return imgs


def vram_mb() -> float:
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_vram():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def benchmark_model(name: str, model_id: str, imgs_all: list) -> dict | None:
    """Load model, run warm-up + timed passes, return result dict."""
    print(f"  Loading {name} ({model_id}) …", end=" ", flush=True)

    try:
        model = YOLO(model_id)
        model.to(DEVICE)
        if HALF:
            model.half()
    except Exception as e:
        print(f"FAILED ({e})")
        return None

    # Count parameters
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Model file size (cached in ultralytics weights dir)
    try:
        from ultralytics.utils import WEIGHTS_DIR
        pt_path = WEIGHTS_DIR / model_id
        model_mb = pt_path.stat().st_size / 1024 ** 2 if pt_path.exists() else 0.0
    except Exception:
        model_mb = 0.0

    warmup_imgs = imgs_all[: N_WARMUP]
    bench_imgs  = imgs_all[: N_BENCH]

    # ── Warm-up ──────────────────────────────────────────────────────────────
    try:
        for i in range(0, len(warmup_imgs), BATCH_SIZE):
            batch = warmup_imgs[i : i + BATCH_SIZE]
            with torch.no_grad():
                model.predict(batch, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
    except Exception as e:
        print(f"WARM-UP FAILED ({e})")
        del model
        gc.collect()
        reset_vram()
        return None

    reset_vram()

    # ── Timed inference ───────────────────────────────────────────────────────
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_processed = 0

    try:
        for i in range(0, len(bench_imgs), BATCH_SIZE):
            batch = bench_imgs[i : i + BATCH_SIZE]
            with torch.no_grad():
                model.predict(batch, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
            n_processed += len(batch)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    except Exception as e:
        print(f"BENCH FAILED ({e})")
        del model
        gc.collect()
        reset_vram()
        return None

    peak_vram = vram_mb()
    ms_per_img = (elapsed / n_processed) * 1000
    fps        = n_processed / elapsed

    print(f"✓  {fps:6.1f} FPS  |  {ms_per_img:5.1f} ms/img  |  "
          f"{peak_vram:6.0f} MB VRAM  |  {n_params:5.1f}M params")

    del model
    gc.collect()
    reset_vram()

    return {
        "name":      name,
        "fps":       fps,
        "ms":        ms_per_img,
        "vram_mb":   peak_vram,
        "params_m":  n_params,
        "size_mb":   model_mb,
    }


def fmt_eta(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def print_table(results: list[dict]):
    W   = 92
    SEP = "─" * W
    HDR = (f"{'Model':<12} {'FPS':>8} {'ms/img':>8} {'VRAM MB':>9} "
           f"{'Params M':>10} {'ETA (100ep)':>13} {'Rank':>5}")

    # Sort by FPS descending
    results_sorted = sorted(results, key=lambda r: r["fps"], reverse=True)

    print("\n")
    print("═" * W)
    print(f"  YOLO26 BENCHMARK  —  RTX 4060 Laptop  /  512×512 tiles  /  "
          f"{TRAIN_IMAGES:,} training images")
    print("═" * W)
    print(HDR)
    print(SEP)

    for rank, r in enumerate(results_sorted, 1):
        # Training FPS ≈ inference FPS / backprop factor
        train_fps      = r["fps"] / BACKPROP_FACTOR
        secs_per_epoch = TRAIN_IMAGES / train_fps
        total_secs     = secs_per_epoch * TRAIN_EPOCHS
        eta_str        = fmt_eta(total_secs)
        flag = " ◄ fastest" if rank == 1 else ""
        print(f"{r['name']:<12} {r['fps']:>8.1f} {r['ms']:>8.1f} "
              f"{r['vram_mb']:>9.0f} {r['params_m']:>10.1f} "
              f"{eta_str:>13} {rank:>5}{flag}")

    print(SEP)
    print(f"  ETA assumes {TRAIN_EPOCHS} epochs, {TRAIN_IMAGES:,} imgs, "
          f"backprop overhead ×{BACKPROP_FACTOR:.0f} vs inference FPS")
    print(SEP)

    # Recommendation
    best      = results_sorted[0]
    threshold = best["fps"] / 4
    balanced  = [r for r in results_sorted if r["fps"] >= threshold]
    recommend = max(balanced, key=lambda r: r["params_m"])

    best_train_secs = (TRAIN_IMAGES / (best["fps"] / BACKPROP_FACTOR)) * TRAIN_EPOCHS
    rec_train_secs  = (TRAIN_IMAGES / (recommend["fps"] / BACKPROP_FACTOR)) * TRAIN_EPOCHS

    print(f"\n  ⚡  Fastest         : {best['name']:8s}  "
          f"{best['fps']:.1f} FPS  →  {fmt_eta(best_train_secs)} to train")
    print(f"  🎯  Best balance    : {recommend['name']:8s}  "
          f"{recommend['fps']:.1f} FPS  →  {fmt_eta(rec_train_secs)} to train  "
          f"({recommend['params_m']:.1f}M params)")
    print(f"\n  Tip: For training on 8 GB VRAM at 512 px, models with")
    print(f"       < 6,000 MB VRAM at inference are generally safe.")
    print("═" * 78)


def main():
    if DEVICE == "cuda":
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[INFO] GPU    : {gpu}  ({vram:.1f} GB VRAM)")
        print(f"[INFO] Mode   : {'FP16' if HALF else 'FP32'}")
    else:
        print("[WARN] No CUDA — running on CPU, results will be slow")

    print(f"[INFO] Batch  : {BATCH_SIZE}  |  Bench images: {N_BENCH}  |  Warmup: {N_WARMUP}")
    print(f"[INFO] Loading {N_WARMUP + N_BENCH} sample images …")

    imgs = sample_images(IMAGE_DIR, N_WARMUP + N_BENCH, SEED)
    if not imgs:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")
    print(f"[INFO] Loaded {len(imgs)} images\n")

    print(f"Benchmarking {len(MODELS)} models:\n")
    results = []

    for name, model_id in MODELS:
        result = benchmark_model(name, model_id, imgs)
        if result:
            results.append(result)

    if results:
        print_table(results)
    else:
        print("[ERROR] No models benchmarked successfully.")


if __name__ == "__main__":
    main()
