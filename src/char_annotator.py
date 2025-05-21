import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import glob

# === CONFIG ===
IMAGE_DIR = r"C:\Python\license-plate-recognition\data\seg_and_ocr\usimages"
LABEL_DIR = os.path.join(IMAGE_DIR, "labels")
os.makedirs(LABEL_DIR, exist_ok=True)

CHAR_CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_CLASSES)}

MAX_DISPLAY_SIZE = 1000  # max size for window/image display

# === GLOBALS ===
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.[pjP][npNP]*")))
image_files = [f for f in image_files if os.path.isfile(f)]
image_index = 0
annotations = []
canvas_img_id = None
img_cv = None
img_tk = None
display_scale = 1.0
box_preview = None

# === TK SETUP ===
root = tk.Tk()
root.title("Character Annotator")

canvas = tk.Canvas(root, cursor="crosshair")
canvas.pack(fill=tk.BOTH, expand=True)

btn_next = tk.Button(root, text="Next Image", command=lambda: next_image())
btn_next.pack()

# === FUNCTIONS ===
def get_next_index():
    for i, img_path in enumerate(image_files):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base + ".txt")
        if not os.path.exists(label_path):
            return i
    return len(image_files)  # done

def load_image(index):
    global img_cv, img_tk, display_scale, canvas_img_id
    canvas.delete("all")
    annotations.clear()

    img_path = image_files[index]
    img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img_cv.shape[:2]

    scale = min(MAX_DISPLAY_SIZE / w, MAX_DISPLAY_SIZE / h, 1.0)
    display_scale = scale
    resized = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
    img_pil = Image.fromarray(resized)
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas.config(width=img_tk.width(), height=img_tk.height())
    canvas_img_id = canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

def save_annotations():
    if not annotations:
        return
    base = os.path.splitext(os.path.basename(image_files[image_index]))[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    h, w = img_cv.shape[:2]
    lines = []
    for x1, y1, x2, y2, char in annotations:
        # Rescale to original image coordinates
        x1, y1, x2, y2 = [int(p / display_scale) for p in [x1, y1, x2, y2]]
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        class_id = CHAR_TO_ID[char.upper()]
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

def next_image():
    global image_index
    save_annotations()
    image_index += 1
    if image_index >= len(image_files):
        messagebox.showinfo("Done", "All images annotated.")
        root.quit()
    else:
        load_image(image_index)

def on_mouse_down(event):
    canvas.start_x = event.x
    canvas.start_y = event.y
    global box_preview
    box_preview = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="green", width=2)

def on_mouse_move(event):
    global box_preview
    if box_preview:
        canvas.coords(box_preview, canvas.start_x, canvas.start_y, event.x, event.y)

def on_mouse_up(event):
    global box_preview
    x1, y1 = canvas.start_x, canvas.start_y
    x2, y2 = event.x, event.y
    if box_preview:
        canvas.delete(box_preview)
        box_preview = None
    canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    char = simpledialog.askstring("Character", "Enter character (A-Z, 0-9):")
    if char and char.upper() in CHAR_CLASSES:
        annotations.append((x1, y1, x2, y2, char.upper()))
    else:
        messagebox.showwarning("Invalid", "Invalid character. Box discarded.")

# === BIND EVENTS ===
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

# === RUN ===
image_index = get_next_index()
if image_index < len(image_files):
    load_image(image_index)
else:
    messagebox.showinfo("Done", "All images already annotated.")
    root.quit()

root.mainloop()
