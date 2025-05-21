import pandas as pd
import cv2
import os
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = r"C:\Python\license-plate-recognition\runs\detect\char_detector\weights\best.pt"
IMAGE_DIR = r"C:\Python\license-plate-recognition\data\seg_and_ocr\usimages"
CSV_PATH = r"C:\Python\license-plate-recognition\data\seg_and_ocr\results.csv"
CONF_THRESHOLD = 0.25

CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# === Load Model and Data ===
model = YOLO(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

total = 0
correct = 0

print("\n=== Plate Evaluation ===\n")

for _, row in df.iterrows():
    filename, gt_text = row['filename'], str(row['text']).strip()
    img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {filename}")
        continue

    img = cv2.imread(img_path)
    results = model(img, conf=CONF_THRESHOLD)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        detections.append((center_x, CLASS_NAMES[cls_id]))

    # Sort left to right
    detections.sort(key=lambda x: x[0])
    pred_text = ''.join([c for _, c in detections])

    is_correct = (pred_text == gt_text)
    if is_correct:
        correct += 1
    total += 1

    print(f"{filename:<20} GT: {gt_text:<10} PRED: {pred_text:<10} {'✅' if is_correct else '❌'}")

# === Final Report ===
acc = (correct / total * 100) if total > 0 else 0
print(f"\n🔍 Total: {total} | ✅ Correct: {correct} | 📊 Accuracy: {acc:.2f}%")
