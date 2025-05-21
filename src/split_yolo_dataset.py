import os
import shutil
import random

# === CONFIG ===
SOURCE_DIR = r"C:\Python\license-plate-recognition\data\seg_and_ocr\usimages"
DEST_DIR = r"C:\Python\license-plate-recognition\char_dataset"
SPLIT_RATIO = 0.9  # 90% train, 10% val
SEED = 42

# === Create destination folders ===
for split in ['train', 'val']:
    os.makedirs(os.path.join(DEST_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'labels', split), exist_ok=True)

# === Collect all .png image files that have matching .txt label ===
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.png')]
paired = [f for f in all_images if os.path.exists(os.path.join(SOURCE_DIR, f.replace('.png', '.txt')))]

print(f"✅ Found {len(paired)} image-label pairs.")

# === Shuffle and split ===
random.seed(SEED)
random.shuffle(paired)

split_index = int(len(paired) * SPLIT_RATIO)
train_files = paired[:split_index]
val_files = paired[split_index:]

def copy_files(file_list, split):
    for fname in file_list:
        base = os.path.splitext(fname)[0]
        img_src = os.path.join(SOURCE_DIR, fname)
        label_src = os.path.join(SOURCE_DIR, base + '.txt')

        img_dst = os.path.join(DEST_DIR, 'images', split, fname)
        label_dst = os.path.join(DEST_DIR, 'labels', split, base + '.txt')

        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(label_src, label_dst)

copy_files(train_files, 'train')
copy_files(val_files, 'val')

print(f"✅ Split complete:")
print(f"  Train: {len(train_files)} images")
print(f"  Val:   {len(val_files)} images")
print(f"  Files saved to: {DEST_DIR}")
