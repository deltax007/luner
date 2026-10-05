import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import umap
import hdbscan
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TILES_DIR      = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\512_size")
JUNK_DIR       = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\junk")
REVIEW_DIR     = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\cluster_review")
VALIDATE_TILES = True   # scan and drop invalid/corrupted PNG tiles before loading
MOVE_ONLY      = True   # set to True to skip feature extraction and only move files
KEEP_CLUSTER_IDS = [357]  # cluster IDs to keep when MOVE_ONLY is enabled

# ── DINO extraction ────────────────────────────────────────────────────────────
BATCH_SIZE     = 32      # safe for 4060 8GB with ViT-S/8
NUM_WORKERS    = 4       # dataloader workers
DINO_MODEL     = "vits"   # vit_small (fast) or vit_base (more accurate)
DINO_PATCH     = 8             # patch size — 8 gives richer features than 16

# ── UMAP ──────────────────────────────────────────────────────────────────────
UMAP_COMPONENTS = 50    # reduce to 50 dims before HDBSCAN (not 2 — preserves structure)
UMAP_NEIGHBORS  = 30    # higher = more global structure preserved
UMAP_MIN_DIST   = 0.0   # 0.0 = tighter clusters, better for HDBSCAN input

# ── HDBSCAN ───────────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER  = 50    # minimum tiles to form a cluster
HDBSCAN_MIN_SAMPLES  = 10    # controls noise sensitivity

# ── Review grid ───────────────────────────────────────────────────────────────
SAMPLES_PER_CLUSTER = 10     # tiles shown per cluster in the review grid
# ─────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────
class TileDataset(Dataset):
    """
    Loads PNG tiles and applies DINO-compatible preprocessing.
    Converts grayscale → 3-channel (DINO expects RGB).
    """
    def __init__(self, tile_paths):
        self.paths     = tile_paths
        self.transform = transforms.Compose([
            transforms.Resize(224),                  # DINO input size
            transforms.Grayscale(num_output_channels=3),  # gray → RGB
            transforms.ToTensor(),
            transforms.Normalize(                    # ImageNet stats still work
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("L")   # load as grayscale
                tensor = self.transform(img)
        except Exception as exc:
            raise OSError(f"Failed to open image {path}: {exc}") from exc

        return tensor, idx


# ── DINO feature extraction ────────────────────────────────────────────────────
def load_dino_model(device):
    print("📥 Loading DINO ViT model...")
    model = torch.hub.load(
        "facebookresearch/dino:main",
        f"dino_{DINO_MODEL}{DINO_PATCH}",
        pretrained=True
    )
    model.eval()
    model = model.to(device)
    print(f"✅ DINO {DINO_MODEL} patch{DINO_PATCH} loaded on {device}")
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extracts CLS token features from DINO for all tiles.
    CLS token = global image summary, perfect for clustering.
    """
    all_features = []

    for imgs, _ in tqdm(loader, desc="🔬 Extracting DINO features"):
        imgs = imgs.to(device, non_blocking=True)

        # DINO forward → CLS token is first output token
        feats = model(imgs)                      # (B, embed_dim)
        all_features.append(feats.cpu().numpy())

        # ✅ Explicit VRAM cleanup each batch — safe for 8 GB
        del imgs, feats
        torch.cuda.empty_cache()

    return np.vstack(all_features)   # (N, embed_dim)


# ── UMAP ──────────────────────────────────────────────────────────────────────
def filter_valid_tile_paths(tile_paths):
    valid_paths = []
    print(f"\n🔎 Validating {len(tile_paths)} tile files...")
    for path in tqdm(tile_paths, desc="✅ Validating tiles", mininterval=5.0):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
        except Exception as exc:
            print(f"⚠️ Skipping invalid tile: {path} ({exc})")
    return valid_paths


def run_umap(features):
    print(f"\n📐 Running UMAP: {features.shape[1]}D → {UMAP_COMPONENTS}D ...")
    reducer = umap.UMAP(
        n_components   = UMAP_COMPONENTS,
        n_neighbors    = UMAP_NEIGHBORS,
        min_dist       = UMAP_MIN_DIST,
        metric         = "cosine",      # cosine works better than euclidean for ViT features
        random_state   = 42,
        low_memory     = False,         # 16 GB RAM — use fast mode
        verbose        = True
    )
    embedded = reducer.fit_transform(features)
    print(f"✅ UMAP done → shape {embedded.shape}")
    return embedded


# ── HDBSCAN ───────────────────────────────────────────────────────────────────
def run_hdbscan(embedded):
    print(f"\n🔍 Running HDBSCAN ...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size  = HDBSCAN_MIN_CLUSTER,
        min_samples       = HDBSCAN_MIN_SAMPLES,
        metric            = "euclidean",
        cluster_selection_method = "eom",   # excess of mass — finds varied cluster sizes
        core_dist_n_jobs  = -1              # use all CPU cores
    )
    labels = clusterer.fit_predict(embedded)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"✅ Found {n_clusters} clusters | {n_noise} noise tiles ({n_noise/len(labels)*100:.1f}%)")
    return labels, clusterer


# ── Cluster review grid ────────────────────────────────────────────────────────
def save_cluster_grid(tile_paths, labels, review_dir):
    """
    Saves a PNG grid of sample tiles for each cluster so you can
    visually decide which clusters are junk.
    """
    review_dir.mkdir(parents=True, exist_ok=True)
    unique_labels = sorted(set(labels))

    print(f"\n🖼  Saving cluster review grids to {review_dir} ...")

    for cluster_id in tqdm(unique_labels, desc="📊 Building grids"):
        tag        = "NOISE" if cluster_id == -1 else f"cluster_{cluster_id:03d}"
        idxs       = np.where(labels == cluster_id)[0]
        sample_idx = np.random.choice(idxs, min(SAMPLES_PER_CLUSTER, len(idxs)), replace=False)

        cols = 5
        rows = (len(sample_idx) + cols - 1) // cols
        fig  = plt.figure(figsize=(cols * 3, rows * 3 + 1))
        fig.suptitle(
            f"{tag}  |  {len(idxs)} tiles",
            fontsize=14, fontweight="bold", y=1.01
        )

        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.05)

        for plot_i, tile_idx in enumerate(sample_idx):
            ax  = fig.add_subplot(gs[plot_i // cols, plot_i % cols])
            img = Image.open(tile_paths[tile_idx]).convert("L")
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

        plt.savefig(review_dir / f"{tag}.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

    print(f"✅ Review grids saved — open {review_dir} to inspect clusters")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_cluster_summary(tile_paths, labels):
    unique_labels = sorted(set(labels))
    print("\n" + "─" * 55)
    print(f"{'CLUSTER':<15} {'TILES':>8}  {'%':>6}  {'TAG'}")
    print("─" * 55)

    total = len(labels)
    for cid in unique_labels:
        count = (labels == cid).sum()
        tag   = "← NOISE (unclustered)" if cid == -1 else ""
        print(f"  cluster {cid:<6}   {count:>8}  {count/total*100:>5.1f}%  {tag}")

    print("─" * 55)
    print(f"  {'TOTAL':<13}   {total:>8}")
    print()
    print("👉 Open REVIEW_DIR, inspect each cluster grid,")
    print("   then re-run with JUNK_CLUSTERS = [0, 3, ...]")
    print("   to move those tiles to the junk folder.\n")


# ── Junk mover ────────────────────────────────────────────────────────────────
def move_junk_tiles(tile_paths, labels, junk_cluster_ids):
    """
    Call this after reviewing the grids.
    Pass the cluster IDs you identified as junk.
    Example: move_junk_tiles(tile_paths, labels, junk_cluster_ids=[0, 2])
    """
    JUNK_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0

    for cid in junk_cluster_ids:
        idxs = np.where(labels == cid)[0]
        for i in tqdm(idxs, desc=f"🗑  Moving cluster {cid}"):
            src = tile_paths[i]
            dst = JUNK_DIR / src.name
            shutil.move(str(src), str(dst))
            moved += 1

    print(f"\n✅ Moved {moved} junk tiles to {JUNK_DIR}")
    return moved


def get_junk_cluster_ids(labels, keep_cluster_ids=None):
    """Return cluster IDs that should be moved to junk."""
    if keep_cluster_ids is None:
        keep_cluster_ids = []
    return [cid for cid in sorted(set(labels)) if cid not in keep_cluster_ids]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Gather tiles ───────────────────────────────────────────────────────
    tile_paths = sorted(TILES_DIR.glob("*.png"))
    print(f"🔍 Found {len(tile_paths)} tiles in {TILES_DIR}\n")

    if not tile_paths:
        print("❌ No tiles found. Check TILES_DIR path.")
        return

    labels_path = REVIEW_DIR / "cluster_labels.npy"
    if MOVE_ONLY:
        if not labels_path.exists():
            print(f"❌ Cannot move files: labels file not found at {labels_path}")
            return

        labels = np.load(labels_path)
        junk_cluster_ids = get_junk_cluster_ids(labels, keep_cluster_ids=KEEP_CLUSTER_IDS)
        print(f"👉 Keeping cluster IDs: {KEEP_CLUSTER_IDS}")
        print(f"👉 Moving {len(junk_cluster_ids)} junk clusters")
        move_junk_tiles(tile_paths, labels, junk_cluster_ids=junk_cluster_ids)
        return

    if VALIDATE_TILES:
        tile_paths = filter_valid_tile_paths(tile_paths)
        print(f"✅ Valid tiles: {len(tile_paths)}\n")

    if not tile_paths:
        print("❌ No tiles found. Check TILES_DIR path.")
        return

    # ── 2. Setup GPU ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("❌ CUDA not available. This script requires a GPU for feasible performance.")
        return

    # ── 3. DataLoader ─────────────────────────────────────────────────────────
    dataset = TileDataset(tile_paths)
    loader  = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,       # must stay ordered for index alignment
        num_workers = NUM_WORKERS,
        pin_memory  = True,        # faster CPU→GPU on 4060
        prefetch_factor = 2
    )

    # ── 4. DINO feature extraction ────────────────────────────────────────────
    model    = load_dino_model(device)
    features = extract_features(model, loader, device)
    del model
    torch.cuda.empty_cache()
    print(f"✅ Features: {features.shape}  (tiles × embed_dim)\n")

    # ── 5. UMAP ───────────────────────────────────────────────────────────────
    embedded = run_umap(features)
    del features   # free RAM — ~(N × 384) floats no longer needed

    # ── 6. HDBSCAN ────────────────────────────────────────────────────────────
    labels, _ = run_hdbscan(embedded)
    del embedded

    # ── 7. Summary + review grids ─────────────────────────────────────────────
    print_cluster_summary(tile_paths, labels)
    save_cluster_grid(tile_paths, labels, REVIEW_DIR)

    # ── 8. Save labels for later use ──────────────────────────────────────────
    labels_path = REVIEW_DIR / "cluster_labels.npy"
    np.save(labels_path, labels)
    print(f"💾 Labels saved to {labels_path}")
    print("   (load with np.load to re-use without re-running)")


if __name__ == "__main__":
    main()