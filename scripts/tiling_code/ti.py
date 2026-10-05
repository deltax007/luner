import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_DIR = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\ohrc_images")
OUT_DIR = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\512_size")

TILE_SIZE = 512
DTYPE = np.uint8

# Toggle preprocessing
APPLY_DESTRIPING = True
# ─────────────────────────────────────────────


# 🔹 Read dimensions from XML
def get_image_shape(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines, samples = None, None

        for axis_array in root.iter('{http://pds.nasa.gov/pds4/pds/v1}Axis_Array'):
            axis_name = axis_array.find('{http://pds.nasa.gov/pds4/pds/v1}axis_name')
            elements = axis_array.find('{http://pds.nasa.gov/pds4/pds/v1}elements')

            if axis_name is not None and elements is not None:
                if axis_name.text == "Line":
                    lines = int(elements.text)
                elif axis_name.text == "Sample":
                    samples = int(elements.text)

        return lines, samples

    except Exception as e:
        print(f"❌ XML error: {xml_path.name} → {e}")
        return None, None


# 🔹 Light destriping (median-based)
def remove_stripes_median(img):
    col_median = np.median(img, axis=0)
    global_median = np.median(col_median)

    corrected = img - col_median + global_median
    return corrected


# 🔹 Global normalization (IMPORTANT)
def normalize_image(img):
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)

    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
        img = (img * 255).astype(np.uint8)

    return img


# 🔹 Tiling function (512×512)
def tile_image(img, tile_size, save_dir, base_name):
    h, w = img.shape
    count = 0

    save_dir.mkdir(parents=True, exist_ok=True)

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):

            tile = img[y:y+tile_size, x:x+tile_size]

            filename = f"{base_name}_y{y}_x{x}.png"
            filepath = save_dir / filename

            # ✅ Direct save (NO per-tile normalization)
            Image.fromarray(tile).save(filepath)

            count += 1

    return count


# 🔹 Process one image
def process_image(img_path):
    xml_path = img_path.with_suffix(".xml")

    if not xml_path.exists():
        print(f"⚠️ Missing XML: {img_path.name}")
        return

    print(f"\n📂 Processing: {img_path.name}")

    height, width = get_image_shape(xml_path)

    if height is None or width is None:
        print("❌ Failed to read shape")
        return

    try:
        img = np.fromfile(img_path, dtype=DTYPE)

        if img.size != height * width:
            print(f"❌ Size mismatch: Expected {height*width}, Got {img.size}")
            return

        img = img.reshape((height, width))

        print(f"🔍 RAW → min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")

        # 🔹 Light destriping
        if APPLY_DESTRIPING:
            img = remove_stripes_median(img)

        # 🔹 Normalize ONCE globally
        img = normalize_image(img)

    except Exception as e:
        print(f"❌ Read error: {e}")
        return

    base_name = img_path.stem

    # 🔹 Tile into 512×512
    count = tile_image(img, TILE_SIZE, OUT_DIR, base_name)

    print(f"✅ Created {count} tiles ({TILE_SIZE}x{TILE_SIZE})")


# 🔹 Main
def main():
    img_files = list(IMG_DIR.glob("*.img"))

    print(f"🔍 Found {len(img_files)} images")

    for img_path in img_files:
        process_image(img_path)
        


if __name__ == "__main__":
    main()