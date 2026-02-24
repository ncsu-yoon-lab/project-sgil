import deepforest.main as df_main
import cv2
import os
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS (tune these)
# ============================================================

# Input
IMAGE_PATH = "../dataset/satellite_images/RaleighSatellite.png"

# DeepForest inference
SCORE_THRESH = 0.12           # model confidence threshold (lower = more boxes)
PATCH_SIZE = 900              # tile size in pixels for predict_tile (bigger => fewer split crowns)
PATCH_OVERLAP = 0.45          # tile overlap fraction (higher => fewer patch-boundary fragments)

# Multi-pass preprocessing (helps dark crowns)
GAMMAS = [1.0, 0.75, .65]     # gamma correction passes (smaller = brighten shadows)
USE_CLAHE = True              # apply CLAHE on L channel to boost local contrast
CLAHE_CLIP_LIMIT = 2.0        # CLAHE strength (higher = stronger contrast; too high can add noise)
CLAHE_TILE_GRID = (8, 8)      # CLAHE tile size (smaller tiles = more local contrast)

# Merge / de-dup
DO_MERGE = True               # cluster + fuse related boxes
                              # (helps big trees collapsing from many small boxes)
MERGE_MODE = "wbf"            # "wbf" (tighter) or "union" (one big envelope)
MERGE_IOU = 0.15              # merge boxes if IoU is at least this
MERGE_IOA = 0.45              # merge boxes if intersection-over-smaller-area is at least this
                              # (catches fragments)
MERGE_CENTER_FRAC = 0.30      # merge boxes if centers are this close
                              # (fraction of the larger box max side)

# NMS merge (final de-dup after fusion)
DO_NMS = True                 # de-duplicate overlapping detections after filtering/merging
NMS_IOU = 0.30                # IoU threshold for suppression

# --- Filtering: size (requested) ---
MIN_SIDE_PX = 25              # reject tiny boxes; each side must be at least this many pixels
MIN_AREA_PX = 500             # reject very small boxes by area (w*h)
MAX_SIDE_PX = 5000            # reject absurdly large boxes (usually false positives)
MAX_AREA_PX = 25_500_000       # reject absurdly large boxes by area

# Optional: fragment/sliver rejection
USE_ASPECT_FILTER = True      # reject extremely skinny boxes (often fragments)
MAX_ASPECT = 3.5              # max(w/h, h/w)

# --- Filtering: color/vegetation ---
# HSV green-ish range
LOWER_GREEN = np.array([
    18, 10, 5
])  # lower HSV bound for "green" pixels (H,S,V) (looser => catches darker green)
UPPER_GREEN = np.array([98, 255, 255])  # upper HSV bound for "green" pixels (H,S,V)

# Shadow-tolerant green test:
BRIGHT_FRACTION = 0.65         # evaluate brightest fraction of pixels in the box (higher => includes more dark canopy)
GREEN_RATIO_THRESH = 0.05      # min fraction of (bright-ish) pixels that must be green

# Optional: "shadow green" rescue (for dark crowns that fail HSV green ratio)
ENABLE_SHADOW_GREEN_RESCUE = True   # allow dark-but-colorful vegetation through
MIN_BRIGHT_V_PCT = 22               # minimum V percentile threshold for "bright" pixels (lower => tolerate shadows)
MIN_LAB_B_BRIGHT = 130.0            # mean LAB b* on bright pixels; >128 biases yellow/green
MIN_LAB_CHROMA_RESCUE = 12.0        # require higher chroma when using rescue path

# Gray/roof rejection (computed on the same bright pixels)
MIN_MEAN_SAT_BRIGHT = 18       # min mean HSV saturation (bright pixels); low = gray/roof
MIN_LAB_CHROMA_BRIGHT = 10     # min mean LAB chroma (bright pixels); low = neutral surfaces

# Vegetation index (Excess Green) on bright pixels: exg = 2G - R - B (on [0..1] channels)
MIN_EXG_BRIGHT = 0.015          # min mean ExG (bright pixels)
                               # low/negative => non-veg

MIN_BRIGHT_PIXELS = 120

# Extra dark-canopy rescue: allow low saturation/chroma boxes if ExG is strong.
# (This helps keep very dark canopies that still show strong green dominance.)
ENABLE_DARK_VEG_RESCUE = True  # let very dark crowns through if ExG says "vegetation"
DARK_VEG_MIN_EXG = 0.06        # strong green dominance even if saturation is low
DARK_VEG_MIN_CHROMA = 6.0      # still require some chroma (avoid pure gray surfaces)

# Drawing
DRAW_REJECTED = True           # draw rejected detections (blue) + rejection reason labels
HIDE_BLUE = False              # if True, never draw rejected (blue) boxes even if DRAW_REJECTED=True
DRAW_THICKNESS_KEEP = 3        # thickness of kept (red) boxes
DRAW_THICKNESS_REJECT = 2      # thickness of rejected boxes
COLOR_KEEP_BRG = (0, 0, 255)   # kept box color (B,G,R)
COLOR_REJECT_BRG = (255, 0, 0)  # rejected box color (B,G,R)

# Output files
SAVE_DEBUG_PASSES = False      # keep the temporary gamma/CLAHE images on disk (debugging)

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


def ioa_smaller(boxA, boxB) -> float:
    """Intersection-over-area-of-smaller-box. Useful for merging fragments inside a big tree box."""
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
    denom = max(1.0, float(min(areaA, areaB)))
    return float(inter) / denom


def center_dist(boxA, boxB) -> float:
    ax = 0.5 * (boxA[0] + boxA[2])
    ay = 0.5 * (
        boxA[1] + boxA[3]
    )
    bx = 0.5 * (boxB[0] + boxB[2])
    by = 0.5 * (boxB[1] + boxB[3])
    return float(np.hypot(ax - bx, ay - by))


def max_side(box) -> float:
    return float(max(1.0, (box[2] - box[0]), (box[3] - box[1])))


def should_merge(boxA, boxB) -> bool:
    if iou(boxA, boxB) >= MERGE_IOU:
        return True
    if ioa_smaller(boxA, boxB) >= MERGE_IOA:
        return True
    # If they’re really close (often patch splits), merge.
    if center_dist(boxA, boxB) <= MERGE_CENTER_FRAC * max(max_side(boxA), max_side(boxB)):
        return True
    return False


def fuse_cluster(boxes: list[tuple[int, int, int, int]], scores: list[float]) -> tuple[int, int, int, int]:
    """Fuse a cluster of boxes into one (either weighted average or union)."""
    if not boxes:
        return (0, 0, 0, 0)

    if MERGE_MODE.lower() == "union" or len(boxes) == 1:
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return (int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2)))

    # Weighted Box Fusion (simple): coordinate = sum(score * coord) / sum(score)
    wts = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(wts).all() or wts.sum() <= 1e-9:
        wts = np.ones(len(boxes), dtype=np.float32)
    wts = wts / (wts.sum() + 1e-9)

    arr = np.asarray(boxes, dtype=np.float32)
    fused = (arr * wts[:, None]).sum(axis=0)
    return (int(fused[0]), int(fused[1]), int(fused[2]), int(fused[3]))


def cluster_and_fuse(boxes: list[tuple[int, int, int, int]], scores: list[float]):
    """Group boxes that likely refer to the same crown, then fuse each group into one box."""
    n = len(boxes)
    if n <= 1:
        return boxes, scores

    # Union-Find for clustering
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(boxes[i], boxes[j]):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    fused_boxes: list[tuple[int, int, int, int]] = []
    fused_scores: list[float] = []
    for idxs in clusters.values():
        b = [boxes[k] for k in idxs]
        s = [scores[k] for k in idxs]
        fused_boxes.append(fuse_cluster(b, s))
        fused_scores.append(float(max(s)))

    return fused_boxes, fused_scores


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


def bright_v_threshold(box_hsv: np.ndarray, bright_fraction: float) -> float:
    """Return the V threshold used to create the bright-pixel mask (percentile of V)."""
    if box_hsv.size == 0:
        return 0.0
    V = box_hsv[:, :, 2].astype(np.uint8).reshape(-1)
    if V.size == 0:
        return 0.0
    return float(np.quantile(V, 1.0 - bright_fraction))


def lab_chroma_exg_b_and_exg_bright(
    box_bgr: np.ndarray, bright_mask: np.ndarray
) -> tuple[float, float, float]:
    """Compute (LAB chroma, ExG, LAB b*) on *bright pixels* only."""
    if box_bgr.size == 0:
        return 0.0, 0.0, 128.0

    lab = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32) - 128.0
    b_ch = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = cv2.magnitude(a_ch, b_ch)
    mean_chroma = float(cv2.mean(chroma, mask=bright_mask)[0])

    mean_lab_b = float(cv2.mean(lab[:, :, 2].astype(np.float32), mask=bright_mask)[0])

    bgr = box_bgr.astype(np.float32) / 255.0
    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]
    exg = 2.0 * G - R - B
    mean_exg = float(cv2.mean(exg, mask=bright_mask)[0])

    return mean_chroma, mean_exg, mean_lab_b


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

    # Filter + collect for merge/NMS
    kept_rows = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_scores: list[float] = []

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
        if USE_ASPECT_FILTER:
            aspect = max(bw / float(bh + 1e-9), bh / float(bw + 1e-9))
            if aspect > MAX_ASPECT:
                rejected.append(((xmin, ymin, xmax, ymax), "too_skinny"))
                continue

        # --- color filters ---
        box_hsv = hsv_img[ymin:ymax, xmin:xmax]
        green_ratio, mean_sat_bright, bright_cnt, bright_mask = green_ratio_and_sat(box_hsv)

        if bright_cnt < MIN_BRIGHT_PIXELS or bright_mask is None:
            rejected.append(((xmin, ymin, xmax, ymax), "too_few_bright_px"))
            continue

        box_bgr = img_bgr[ymin:ymax, xmin:xmax]
        mean_chroma, mean_exg, mean_lab_b = lab_chroma_exg_b_and_exg_bright(box_bgr, bright_mask)

        # Primary gray-roof kill-switch: low saturation OR low chroma on bright pixels
        if mean_sat_bright < MIN_MEAN_SAT_BRIGHT or mean_chroma < MIN_LAB_CHROMA_BRIGHT:
            # Dark canopy rescue: some trees are very dark but still have strong "green dominance".
            if ENABLE_DARK_VEG_RESCUE and mean_exg >= DARK_VEG_MIN_EXG and mean_chroma >= DARK_VEG_MIN_CHROMA:
                pass
            else:
                rejected.append(((xmin, ymin, xmax, ymax), "low_colorfulness"))
                continue

        # Secondary vegetation proxy: ExG must be at least mildly positive
        if mean_exg < MIN_EXG_BRIGHT:
            rejected.append(((xmin, ymin, xmax, ymax), "low_exg"))
            continue

        # Main green test; if it fails, optionally allow a "shadow green" rescue
        if green_ratio < GREEN_RATIO_THRESH:
            if not ENABLE_SHADOW_GREEN_RESCUE:
                rejected.append(((xmin, ymin, xmax, ymax), "low_green_ratio"))
                continue

            v_thr = bright_v_threshold(box_hsv, BRIGHT_FRACTION)
            if v_thr < MIN_BRIGHT_V_PCT:
                rejected.append(((xmin, ymin, xmax, ymax), "low_green_ratio"))
                continue

            # Rescue rule: dark crowns often have weak HSV green ratio, but still show
            # (a) decent colorfulness and (b) b* shifted above neutral (128).
            if not (mean_lab_b >= MIN_LAB_B_BRIGHT and mean_chroma >= MIN_LAB_CHROMA_RESCUE):
                rejected.append(((xmin, ymin, xmax, ymax), "low_green_ratio"))
                continue

        # Keep
        r = row.copy()
        r["green_ratio_bright"] = green_ratio
        r["mean_sat_bright"] = mean_sat_bright
        r["lab_chroma_bright"] = mean_chroma
        r["lab_b_bright"] = mean_lab_b
        r["exg_bright"] = mean_exg
        r["bright_px"] = bright_cnt

        kept_rows.append(r)
        kept_boxes.append((xmin, ymin, xmax, ymax))
        kept_scores.append(float(row["score"]) if "score" in row else 1.0)

    if not kept_rows:
        pd.DataFrame(pred_all.head(0)).to_csv(filt_csv_path, index=False)
        raise RuntimeError("All predictions filtered out. Lower thresholds or size filters.")

    # Cluster + fuse (big-tree fragmentation fix)
    merged_boxes = kept_boxes
    merged_scores = kept_scores
    if DO_MERGE and len(kept_boxes) > 1:
        merged_boxes, merged_scores = cluster_and_fuse(kept_boxes, kept_scores)

    # NMS on merged
    keep_idx = list(range(len(merged_boxes)))
    if DO_NMS and len(keep_idx) > 1:
        keep_idx = nms(np.array(merged_boxes), np.array(merged_scores), iou_thresh=NMS_IOU)

    # For CSV: we don’t have per-fused-row metadata; keep the filtered original rows.
    kept_df = pd.DataFrame(kept_rows)
    kept_df.to_csv(filt_csv_path, index=False)

    # Draw
    annotated = img_bgr.copy()

    # Draw rejected (blue) first
    if DRAW_REJECTED and not HIDE_BLUE:
        for (x1, y1, x2, y2), reason in rejected:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_REJECT_BRG, DRAW_THICKNESS_REJECT)
            cv2.putText(
                annotated, reason, (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_REJECT_BRG, 1, cv2.LINE_AA
            )

    # Draw kept (red) on top
    for i in keep_idx:
        x1, y1, x2, y2 = merged_boxes[i]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_KEEP_BRG, DRAW_THICKNESS_KEEP)

    ok = cv2.imwrite(out_path, annotated)
    if not ok:
        raise RuntimeError(f"Failed to write output image to: {out_path}")

    # Summary
    print(f"Saved annotated image to: {out_path}")
    print(f"Saved raw predictions to: {raw_csv_path}")
    print(f"Saved filtered predictions to: {filt_csv_path}")
    print(f"Total raw detections (all passes): {len(pred_all)}")
    print(f"Kept pre-merge: {len(kept_rows)}")
    if DO_MERGE:
        print(f"After merge: {len(merged_boxes)}")
    print(f"Kept after NMS: {len(keep_idx)}")
    print(f"Rejected (blue): {len(rejected)}")


if __name__ == "__main__":
    main()
