from pathlib import Path
import shutil

EXCT_DIR = Path(r"G:\My Drive\pradan-ohrc-dataset\pradan.issdc.gov.in\ch2\protected\downloadData\POST_OD\isda_archive\ch2_bundle\cho_bundle\nop\ohr_collection\data\exct")

DEST_DIR = Path(r"G:\My Drive\pradan-ohrc-dataset\pradan.issdc.gov.in\ch2\protected\downloadData\POST_OD\isda_archive\ch2_bundle\cho_bundle\nop\ohr_collection\data\coc\dataset\ohrc_images")

EXTENSIONS = {".img", ".xml"}   # file types to copy

DEST_DIR.mkdir(parents=True, exist_ok=True)

img_count = 0
xml_count = 0

print("🚀 Starting copy...\n")

data_dirs = list(EXCT_DIR.rglob("data"))
print(f"Found {len(data_dirs)} 'data' directories")

for data_dir in data_dirs:
    if data_dir.is_dir():
        print(f"Processing data dir: {data_dir}")
        files_found = list(data_dir.rglob("*"))
        print(f"  Found {len(files_found)} total items in {data_dir}")
        for file in files_found:
            if file.is_file() and file.suffix.lower() in EXTENSIONS:
                dst = DEST_DIR / file.name
                print(f"  Copying {file} to {dst}")
                shutil.copy2(file, dst)
                
                if file.suffix.lower() == ".img":
                    img_count += 1
                elif file.suffix.lower() == ".xml":
                    xml_count += 1
                
                print(f"✅ Copied: {file.name}")

print("\n📊 RESULTS:")
print(f"IMG: {img_count}")
print(f"XML: {xml_count}")