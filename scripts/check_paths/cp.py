import os

base_path = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\ohrc_images"

dest_path = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\tiles"
print("🔍 Checking destination folder...")
if os.path.exists(dest_path):
    print("✅ Destination exists")
else:
    print("❌ Destination NOT found")

print("\n🔍 Searching for IMG/XML files in base folder...\n")

found_img = 0
found_xml = 0

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".img"):
            found_img += 1
            print(f"  ✓ Found IMG: {file}")
        if file.endswith(".xml"):
            found_xml += 1
            print(f"  ✓ Found XML: {file}")

print("\n📊 RESULTS:")
print(f"IMG files found: {found_img}")
print(f"XML files found: {found_xml}")