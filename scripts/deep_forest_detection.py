import deepforest.main as df_main
import cv2
import os
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS (tune these)
# ============================================================

# Input
IMAGE_PATH = "../dataset/satellite_images/RaleighSatelliteSuperCropped.png"

# DeepForest inference
SCORE_THRESH = 0.12
PATCH_SIZE = 600
PATCH_OVERLAP = 0.25

# Multi-pass preprocessing (helps dark crowns)
GAMMAS = [1.0, 0.75]         # add 0.65 if needed
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# NMS merge
DO_NMS = True
NMS_IOU = 0.35

# --- Filtering: size (requested) ---
MIN_SIDE_PX = 25             # reject if width or height < this
MIN_AREA_PX = 500            # reject if (w*h) < this (very effective)
MAX_SIDE_PX = 2200           # safety
MAX_AREA_PX = 2_500_000       # safety

# --- Filtering: color/vegetation ---
# HSV green-ish range
LOWER_GREEN = np.array([20, 25, 20])
UPPER_GREEN = np.array([95, 255, 255])

# Shadow-tolerant green test:
BRIGHT_FRACTION = 0.50        # consider only brightest 50% pixels by V
GREEN_RATIO_THRESH = 0.06     # require >= this fraction of (bright pixels) to be green

# Gray/roof rejection (computed on the same bright pixels)
MIN_MEAN_SAT_BRIGHT = 28      # if the brightest pixels are basically unsaturated, reject
MIN_LAB_CHROMA_BRIGHT = 12   # mean sqrt((a-128)^2+(b-128)^2) over bright pixels

# Vegetation index (Excess Green) on bright pixels: exg = 2G - R - B (on [0..1] channels)
MIN_EXG_BRIGHT = 0.02

MIN_BRIGHT_PIXELS = 150       # if a box has too few "bright pixels", ratio tests become unstable

# Drawing
DRAW_THICKNESS_KEEP = 3
DRAW_THICKNESS_REJECT = 2
COLOR_KEEP_BRG = (0, 0, 255)      # red
COLOR_REJECT_BRG = (255, 0, 0)    # blue

# Output files
SAVE_DEBUG_PASSES = False         # set True if you want to keep the gamma/clahe temp images

# ============================================================
# End constants
# ============================================================


def apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_bgr
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_clahe_lab(img_bgr: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-9)


def nms(boxes, scores, iou_thresh=0.35):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        rest = idxs[1:]
        survivors = []
        for r in rest:
            if iou(boxes[cur], boxes[r]) <= iou_thresh:
                survivors.append(r)
        idxs = np.array(survivors, dtype=int)
    return keep


def bright_pixel_masks(box_hsv: np.ndarray, bright_fraction: float):
    """
    Returns:
      bright_mask (uint8 0/255) for brightest fraction of pixels by V channel
      denom = countNonZero(bright_mask)
      bright_S = saturation values within box (uint8)
    """
    V = box_hsv[:, :, 2].astype(np.uint8)
    v_flat = V.reshape(-1)
    if v_flat.size == 0:
        return None, 0

    thresh = np.quantile(v_flat, 1.0 - bright_fraction)
    bright_mask = (V >= thresh).astype(np.uint8) * 255
    denom = int(cv2.countNonZero(bright_mask))
    return bright_mask, denom


def lab_chroma_and_exg_bright(box_bgr: np.ndarray, bright_mask: np.ndarray) -> tuple[float, float]:
    """Compute LAB chroma + ExG on *bright pixels* only.

    - LAB chroma is a robust "how colorful" measurement; gray roofs/asphalt will be low.
    - ExG (excess green) is a simple vegetation proxy; gray objects trend near 0.

    Returns: (mean_chroma, mean_exg)
    """
    if box_bgr.size == 0:
        return 0.0, 0.0

    lab = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = cv2.magnitude(a, b)  # sqrt(a^2 + b^2)
    mean_chroma = float(cv2.mean(chroma, mask=bright_mask)[0])

    # ExG on normalized RGB
    bgr = box_bgr.astype(np.float32) / 255.0
    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]
    exg = 2.0 * G - R - B
    mean_exg = float(cv2.mean(exg, mask=bright_mask)[0])

    return mean_chroma, mean_exg


def green_ratio_and_sat(box_hsv: np.ndarray) -> tuple[float, float, int, np.ndarray | None]:
    """
    Shadow-tolerant green ratio computed on brightest pixels only, plus mean saturation
    of those brightest pixels.

    Returns: (green_ratio, mean_sat_bright, bright_pixel_count, bright_mask)
    """
    if box_hsv.size == 0:
        return 0.0, 0.0, 0, None

    bright_mask, bright_count = bright_pixel_masks(box_hsv, BRIGHT_FRACTION)
    if bright_mask is None or bright_count == 0:
        return 0.0, 0.0, 0, None

    green_mask = cv2.inRange(box_hsv, LOWER_GREEN, UPPER_GREEN)
    green_and_bright = cv2.bitwise_and(green_mask, bright_mask)

    green_ratio = float(cv2.countNonZero(green_and_bright)) / float(
        bright_count + 1e-9
    )

    # mean saturation over bright pixels
    S = box_hsv[:, :, 1].astype(np.uint8)
    mean_sat_bright = float(cv2.mean(S, mask=bright_mask)[0])

    return green_ratio, mean_sat_bright, bright_count, bright_mask


def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

    h, w = img_bgr.shape[:2]
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    base, ext = os.path.splitext(IMAGE_PATH)
    work_dir = os.path.dirname(IMAGE_PATH) or "."
    out_path = f"{base}_DEEPFOREST_multipass_boxes.png"
    raw_csv_path = f"{base}_deepforest_multipass_raw.csv"
    filt_csv_path = f"{base}_deepforest_multipass_filtered.csv"

    # Model
    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    model.model.score_thresh = SCORE_THRESH

    # Multi-pass predict
    all_preds = []
    tmp_paths = []

    for gamma in GAMMAS:
        proc = apply_gamma(img_bgr, gamma)
        if USE_CLAHE:
            proc = apply_clahe_lab(
                proc,
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_grid_size=CLAHE_TILE_GRID,
            )

        tmp_path = os.path.join(work_dir, f"__tmp_df_gamma{gamma}_clahe{int(USE_CLAHE)}.jpg")
        cv2.imwrite(tmp_path, proc)
        tmp_paths.append(tmp_path)

        pred = model.predict_tile(
            path=tmp_path,
            patch_size=PATCH_SIZE,
            patch_overlap=PATCH_OVERLAP
        )

        if pred is not None and len(pred) > 0:
            pred = pred.copy()
            pred["gamma"] = gamma
            pred["clahe"] = int(USE_CLAHE)
            all_preds.append(pred)

    if not SAVE_DEBUG_PASSES:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

    if not all_preds:
        raise RuntimeError("No predictions returned from any pass.")

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all.to_csv(raw_csv_path, index=False)

    # Filter + collect for NMS
    kept_rows = []
    kept_boxes = []
    kept_scores = []

    rejected = []  # for blue boxes: (box, reason)

    for _, row in pred_all.iterrows():
        xmin, ymin, xmax, ymax = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)

        # Clamp
        xmin = max(0, min(w - 1, xmin))
        ymin = max(0, min(h - 1, ymin))
        xmax = max(0, min(w - 1, xmax))
        ymax = max(0, min(h - 1, ymax))

        if xmax <= xmin or ymax <= ymin:
            continue

        bw = xmax - xmin
        bh = ymax - ymin
        area = bw * bh

        # --- size filters ---
        if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX:
            rejected.append(((xmin, ymin, xmax, ymax), "too_small_side"))
            continue
        if area < MIN_AREA_PX:
            rejected.append(((xmin, ymin, xmax, ymax), "too_small_area"))
            continue
        if bw > MAX_SIDE_PX or bh > MAX_SIDE_PX or area > MAX_AREA_PX:
            rejected.append(((xmin, ymin, xmax, ymax), "too_large"))
            continue

        # --- color filters ---
        box_hsv = hsv_img[ymin:ymax, xmin:xmax]
        green_ratio, mean_sat_bright, bright_cnt, bright_mask = green_ratio_and_sat(box_hsv)

        if bright_cnt < MIN_BRIGHT_PIXELS or bright_mask is None:
            rejected.append(((xmin, ymin, xmax, ymax), "too_few_bright_px"))
            continue

        box_bgr = img_bgr[ymin:ymax, xmin:xmax]
        mean_chroma, mean_exg = lab_chroma_and_exg_bright(box_bgr, bright_mask)

        # Primary gray-roof kill-switch: low saturation OR low chroma on bright pixels
        if mean_sat_bright < MIN_MEAN_SAT_BRIGHT or mean_chroma < MIN_LAB_CHROMA_BRIGHT:
            rejected.append(((xmin, ymin, xmax, ymax), "low_colorfulness"))
            continue

        # Secondary vegetation proxy: ExG must be at least mildly positive
        if mean_exg < MIN_EXG_BRIGHT:
            rejected.append(((xmin, ymin, xmax, ymax), "low_exg"))
            continue

        if green_ratio < GREEN_RATIO_THRESH:
            rejected.append(((xmin, ymin, xmax, ymax), "low_green_ratio"))
            continue

        # Keep
        r = row.copy()
        r["green_ratio_bright"] = green_ratio
        r["mean_sat_bright"] = mean_sat_bright
        r["lab_chroma_bright"] = mean_chroma
        r["exg_bright"] = mean_exg
        r["bright_px"] = bright_cnt

        kept_rows.append(r)
        kept_boxes.append((xmin, ymin, xmax, ymax))
        kept_scores.append(float(row["score"]) if "score" in row else 1.0)

    if not kept_rows:
        pd.DataFrame(pred_all.head(0)).to_csv(filt_csv_path, index=False)
        raise RuntimeError("All predictions filtered out. Lower thresholds or size filters.")

    # NMS on kept
    keep_idx = list(range(len(kept_rows)))
    if DO_NMS and len(keep_idx) > 1:
        keep_idx = nms(np.array(kept_boxes), np.array(kept_scores), iou_thresh=NMS_IOU)

    kept_df = pd.DataFrame([kept_rows[i] for i in keep_idx])
    kept_df.to_csv(filt_csv_path, index=False)

    # Draw
    annotated = img_bgr.copy()

    # Draw rejected (blue) first
    for (x1, y1, x2, y2), reason in rejected:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_REJECT_BRG, DRAW_THICKNESS_REJECT)
        # small label (optional)
        cv2.putText(
            annotated, reason, (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_REJECT_BRG, 1, cv2.LINE_AA
        )

    # Draw kept (red) on top
    for i in keep_idx:
        x1, y1, x2, y2 = kept_boxes[i]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_KEEP_BRG, DRAW_THICKNESS_KEEP)

    ok = cv2.imwrite(out_path, annotated)
    if not ok:
        raise RuntimeError(f"Failed to write output image to: {out_path}")

    # Summary
    print(f"Saved annotated image to: {out_path}")
    print(f"Saved raw predictions to: {raw_csv_path}")
    print(f"Saved filtered predictions to: {filt_csv_path}")
    print(f"Total raw detections (all passes): {len(pred_all)}")
    print(f"Kept pre-NMS: {len(kept_rows)}")
    print(f"Kept after NMS: {len(keep_idx)}")
    print(f"Rejected (blue): {len(rejected)}")


if __name__ == "__main__":
    main()
