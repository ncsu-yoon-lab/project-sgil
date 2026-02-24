import os
import cv2
import numpy as np
import pandas as pd

image_path = "../dataset/satellite_trees/RaleighSatelliteSuperCropped.png"

base, ext = os.path.splitext(image_path)
out_path = f"{base}_HSV_SIMPLE_boxes.png"
csv_path = f"{base}_HSV_SIMPLE_boxes.csv"
mask_path = f"{base}_HSV_SIMPLE_mask.png"

# --- EXACT-ish settings from the first run ---
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])

KERNEL = np.ones((5, 5), np.uint8)

MIN_W = 10
MIN_H = 10

DRAW_THICKNESS = 2

img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 1) green mask
mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

# 2) cleanup: CLOSE then OPEN
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

cv2.imwrite(mask_path, mask)

# 3) contours -> boxes
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # optional if you display with matplotlib
annotated = img_bgr.copy()

rows = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # filter very small regions (this was the only filter in the first run)
    if w > MIN_W and h > MIN_H:
        x2, y2 = x + w, y + h
        rows.append({"xmin": x, "ymin": y, "xmax": x2, "ymax": y2, "w": w, "h": h})
        cv2.rectangle(annotated, (x, y), (x2, y2), (0, 0, 255), DRAW_THICKNESS)

pd.DataFrame(rows).to_csv(csv_path, index=False)
ok = cv2.imwrite(out_path, annotated)
if not ok:
    raise RuntimeError(f"Failed to write output image to: {out_path}")

print(f"Saved mask to: {mask_path}")
print(f"Saved annotated image to: {out_path}")
print(f"Saved boxes CSV to: {csv_path}")
print(f"Contours found: {len(contours)}")
print(f"Boxes kept: {len(rows)}")