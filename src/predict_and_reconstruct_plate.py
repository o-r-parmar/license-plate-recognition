import cv2
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = r"C:\Python\license-plate-recognition\runs\detect\train3\weights\best.pt"  # update path if different
IMAGE_PATH = r"C:\Python\license-plate-recognition\data\seg_and_ocr\usimages\ca574.png"        # your test plate image

# Class names must match your `data.yaml`
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# === Load model ===
model = YOLO(MODEL_PATH)

# === Run inference ===
results = model(IMAGE_PATH, conf=0.3)[0]  # adjust confidence if needed

# === Extract boxes and sort left-to-right ===
detections = []

for box in results.boxes:
    cls_id = int(box.cls)
    char = CLASS_NAMES[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    center_x = (x1 + x2) // 2
    detections.append((center_x, char))

# Sort by x-position (left to right)
detections.sort(key=lambda x: x[0])
plate_string = ''.join([char for _, char in detections])

# === Show Results ===
print("🔍 Detected Plate:", plate_string)

# Optional: visualize
image = cv2.imread(IMAGE_PATH)
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls)
    label = CLASS_NAMES[cls_id]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.imshow("Detected Plate", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
