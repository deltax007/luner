import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_DIR = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\ohrc_images")
OUT_DIR = Path(r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles\512_size")
DTYPE = np.uint8

# Toggle this if you want stripe removal
REMOVE_STRIPES = False
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


# 🔹 Optional stripe removal
def remove_vertical_stripes(img):
    col_mean = np.mean(img, axis=0)
    return img - col_mean + np.mean(col_mean)


# 🔹 Global normalization (IMPORTANT)
def normalize_image(img):
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)

    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
        img = (img * 255).astype(np.uint8)

    return img


# 🔹 Split image into 5 vertical parts
def split_image(img, save_dir, base_name):
    h, w = img.shape
    num_parts = 5

    part_height = h // num_parts
    remainder = h % num_parts

    save_dir.mkdir(parents=True, exist_ok=True)

    start_y = 0

    for i in range(num_parts):
        extra = 1 if i < remainder else 0
        end_y = start_y + part_height + extra

        part = img[start_y:end_y, :]

        filename = f"{base_name}_part_{i+1}.png"
        filepath = save_dir / filename

        Image.fromarray(part).save(filepath)

        print(f"✅ Saved {filename} ({part.shape[0]} x {part.shape[1]})")

        start_y = end_y


# 🔹 Process image
def process_image(img_path):
    xml_path = img_path.with_suffix(".xml")

    if not xml_path.exists():
        print(f"⚠️ Missing XML: {img_path.name}")
        return

    print(f"\n📂 Processing: {img_path.name}")

    height, width = get_image_shape(xml_path)

    if height is None or width is None:
        print("❌ Could not read shape from XML")
        return

    try:
        img = np.fromfile(img_path, dtype=DTYPE)

        if img.size != height * width:
            print(f"❌ Size mismatch: Expected {height*width}, Got {img.size}")
            return

        img = img.reshape((height, width))

        print(f"🔍 RAW → min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")

        # 🔹 Optional stripe removal
        if REMOVE_STRIPES:
            img = remove_vertical_stripes(img)

        # 🔹 Normalize ONCE
        img = normalize_image(img)

    except Exception as e:
        print(f"❌ Read error: {e}")
        return

    base_name = img_path.stem
    split_image(img, OUT_DIR, base_name)


# 🔹 Main
def main():
    img_path = IMG_DIR / "ch2_ohr_ncp_20250927T0511182400_d_img_d18.img"

    if not img_path.exists():
        print("❌ Image file not found")
        return

    process_image(img_path)


if __name__ == "__main__":
    main()