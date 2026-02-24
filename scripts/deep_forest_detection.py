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

# --- Filtering: size ---
MIN_SIDE_PX = 25              # reject tiny boxes; each side must be at least this many pixels
MIN_AREA_PX = 500             # reject very small boxes by area (w*h)
MAX_SIDE_PX = 5000            # reject absurdly large boxes (usually false positives)
MAX_AREA_PX = 25_500_000       # reject absurdly large boxes by area

# Optional: fragment/sliver rejection
USE_ASPECT_FILTER = True      # reject extremely skinny boxes (often fragments)
MAX_ASPECT = 3.5              # max(w/h, h/w)

# --- Filtering: simple vegetation/color gate ---
# This is intentionally lightweight (no LAB/ExG stats, no tuning script).
# It helps reject non-tree false positives (roofs, shadows, pavement) that DeepForest can pick up.
USE_VEG_FILTER = True

# "Green" hue range in HSV (OpenCV hue: [0..179]). These are deliberately broad.
LOWER_GREEN_HSV = (20, 15, 10)
UPPER_GREEN_HSV = (95, 255, 255)

# We only evaluate the brightest pixels in the box (makes it more robust to shadows).
BRIGHT_FRACTION = 0.50        # fraction of brightest pixels (by V) considered
MIN_GREEN_RATIO = 0.07        # min fraction of bright pixels that fall in green hue range
MIN_MEAN_SAT_BRIGHT = 16.0    # min mean saturation among bright pixels
MIN_MEAN_V_BRIGHT = 40.0      # min mean brightness among bright pixels (helps reject gray roofs)

# Dominant hue among bright pixels must be in green range. Helps reject tan/white roofs.
USE_DOMINANT_HUE_CHECK = True
DOMINANT_HUE_BINS = 36        # number of hue bins over [0..179]
MIN_DOMINANT_HUE_FRAC = 0.16  # fraction of bright pixels falling in the dominant bin

# Drawing
DRAW_REJECTED = True          # draw rejected detections (blue) + rejection reason labels
HIDE_BLUE = False             # if True, never draw rejected (blue) boxes even if DRAW_REJECTED=True
DRAW_THICKNESS_KEEP = 3       # thickness of kept (red) boxes
DRAW_THICKNESS_REJECT = 2     # thickness of rejected boxes
COLOR_KEEP_BRG = (0, 0, 255)  # kept box color (B,G,R)
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


def _bright_pixel_mask_v(box_hsv: np.ndarray, bright_fraction: float) -> np.ndarray | None:
    """Return a mask (uint8 0/255) selecting the brightest pixels by V within a crop."""
    if box_hsv.size == 0:
        return None
    V = box_hsv[:, :, 2].astype(np.uint8)
    if V.size == 0:
        return None
    v_flat = V.reshape(-1)
    # quantile can be slightly expensive but crops are small.
    thresh = float(np.quantile(v_flat, 1.0 - float(bright_fraction)))
    mask = (V >= thresh).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return None
    return mask


def veg_ok(
    crop_hsv: np.ndarray,
    bright_fraction: float,
    lower_green_hsv: tuple[int, int, int],
    upper_green_hsv: tuple[int, int, int],
    min_green_ratio: float,
    min_mean_sat_bright: float,
    min_mean_v_bright: float,
    use_dominant_hue_check: bool,
    dominant_hue_bins: int,
    min_dominant_hue_frac: float,
) -> bool:
    """Lightweight vegetation gate.

    Checks "green ratio" among bright pixels + mean saturation/brightness.
    Optionally enforces that the dominant hue among bright pixels is in green range.
    """
    bright_mask = _bright_pixel_mask_v(crop_hsv, bright_fraction)
    if bright_mask is None:
        return False

    bright_cnt = float(cv2.countNonZero(bright_mask))
    if bright_cnt <= 0:
        return False

    lower = np.array(lower_green_hsv, dtype=np.uint8)
    upper = np.array(upper_green_hsv, dtype=np.uint8)

    # green ratio among bright pixels
    green_mask = cv2.inRange(crop_hsv, lower, upper)
    green_and_bright = cv2.bitwise_and(green_mask, bright_mask)
    green_cnt = float(cv2.countNonZero(green_and_bright))
    green_ratio = green_cnt / (bright_cnt + 1e-9)

    # mean sat/value among bright pixels
    S = crop_hsv[:, :, 1].astype(np.uint8)
    V = crop_hsv[:, :, 2].astype(np.uint8)
    mean_sat_bright = float(cv2.mean(S, mask=bright_mask)[0])
    mean_v_bright = float(cv2.mean(V, mask=bright_mask)[0])

    if green_ratio < float(min_green_ratio):
        return False
    if mean_sat_bright < float(min_mean_sat_bright):
        return False
    if mean_v_bright < float(min_mean_v_bright):
        return False

    if not use_dominant_hue_check:
        return True

    # dominant hue check on bright pixels
    H = crop_hsv[:, :, 0].astype(np.uint8)
    h_vals = H[bright_mask > 0]
    if h_vals.size == 0:
        return False

    bins = int(max(6, dominant_hue_bins))
    # Histogram bins over [0..179]
    hist, bin_edges = np.histogram(h_vals, bins=bins, range=(0, 180))
    dom_idx = int(np.argmax(hist))
    dom_frac = float(hist[dom_idx]) / float(h_vals.size + 1e-9)
    if dom_frac < float(min_dominant_hue_frac):
        # no strong dominant hue => likely mixed material/texture, reject
        return False

    dom_center = 0.5 * (bin_edges[dom_idx] + bin_edges[dom_idx + 1])
    dom_center = int(dom_center)

    # Check if dominant bin center is within green hue bounds
    # (wrap not handled here; green doesn't wrap)
    if not (int(lower_green_hsv[0]) <= dom_center <= int(upper_green_hsv[0])):
        return False

    return True


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


def fuse_cluster(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
) -> tuple[int, int, int, int]:
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
        fused_boxes.append(
            fuse_cluster(b, s)
        )
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


def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h, w = img_bgr.shape[:2]

    base, ext = os.path.splitext(IMAGE_PATH)
    work_dir = os.path.dirname(IMAGE_PATH) or "."
    out_path = f"{base}_DEEPFOREST_multipass_boxes.png"
    raw_csv_path = f"{base}_deepforest_multipass_raw.csv"
    filt_csv_path = f"{base}_deepforest_multipass_filtered.csv"

    # Model
    model = df_main.deepforest()
    # model.load_model(model_name="weecology/deepforest-tree", revision="main")
    model.load_model()
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

        # --- vegetation/color gate ---
        if USE_VEG_FILTER:
            crop_hsv = hsv_img[ymin:ymax, xmin:xmax]
            if not veg_ok(
                crop_hsv,
                bright_fraction=BRIGHT_FRACTION,
                lower_green_hsv=LOWER_GREEN_HSV,
                upper_green_hsv=UPPER_GREEN_HSV,
                min_green_ratio=MIN_GREEN_RATIO,
                min_mean_sat_bright=MIN_MEAN_SAT_BRIGHT,
                min_mean_v_bright=MIN_MEAN_V_BRIGHT,
                use_dominant_hue_check=USE_DOMINANT_HUE_CHECK,
                dominant_hue_bins=DOMINANT_HUE_BINS,
                min_dominant_hue_frac=MIN_DOMINANT_HUE_FRAC,
            ):
                rejected.append(((xmin, ymin, xmax, ymax), "not_green"))
                continue

        # Keep
        kept_rows.append(row.copy())
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
