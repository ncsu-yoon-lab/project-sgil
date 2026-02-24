"""Interactive-ish helper to inspect vegetation/color stats for DeepForest boxes.

Usage (example):
  uv run python scripts/tune_color_thresholds.py \
    --image ../dataset/satellite_images/RaleighSatellite.png

It will print summary stats for each candidate box (from DeepForest raw predictions),
and highlight which filter stage rejected it.

This is meant to help tune thresholds for *darker green* crowns.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

import deepforest.main as df_main


@dataclass
class Stats:
    green_ratio: float
    mean_sat_bright: float
    mean_chroma: float
    mean_exg: float
    mean_lab_b: float
    v_thr: float
    bright_cnt: int


def bright_pixel_masks(box_hsv: np.ndarray, bright_fraction: float):
    V = box_hsv[:, :, 2].astype(np.uint8)
    if V.size == 0:
        return None, 0, 0.0
    v_flat = V.reshape(-1)
    thresh = float(np.quantile(v_flat, 1.0 - bright_fraction))
    bright_mask = (V >= thresh).astype(np.uint8) * 255
    denom = int(cv2.countNonZero(bright_mask))
    return bright_mask, denom, thresh


def lab_chroma_exg_bright(box_bgr: np.ndarray, bright_mask: np.ndarray):
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


def green_ratio_and_sat(box_hsv: np.ndarray, lower_green, upper_green, bright_fraction: float):
    bright_mask, bright_count, v_thr = bright_pixel_masks(box_hsv, bright_fraction)
    if bright_mask is None or bright_count == 0:
        return 0.0, 0.0, 0, None, v_thr

    green_mask = cv2.inRange(box_hsv, lower_green, upper_green)
    green_and_bright = cv2.bitwise_and(green_mask, bright_mask)
    green_ratio = float(cv2.countNonZero(green_and_bright)) / float(bright_count + 1e-9)

    S = box_hsv[:, :, 1].astype(np.uint8)
    mean_sat_bright = float(cv2.mean(S, mask=bright_mask)[0])
    return green_ratio, mean_sat_bright, bright_count, bright_mask, v_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--patch-size", type=int, default=900)
    ap.add_argument("--patch-overlap", type=float, default=0.45)
    ap.add_argument("--bright-fraction", type=float, default=0.50)
    ap.add_argument("--lower-green", type=int, nargs=3, default=[20, 15, 10])
    ap.add_argument("--upper-green", type=int, nargs=3, default=[95, 255, 255])
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    model.model.score_thresh = args.score_thresh

    base, _ = os.path.splitext(args.image)
    tmp_path = base + "__tmp_df.jpg"
    cv2.imwrite(tmp_path, img_bgr)

    pred = model.predict_tile(
        tmp_path,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
    )
    if pred is None or len(pred) == 0:
        print("No predictions")
        return

    lower_green = np.array(args.lower_green)
    upper_green = np.array(args.upper_green)

    rows = []
    for _, row in pred.iterrows():
        xmin, ymin, xmax, ymax = map(int, (row.xmin, row.ymin, row.xmax, row.ymax))
        box_hsv = hsv_img[ymin:ymax, xmin:xmax]
        green_ratio, mean_sat_bright, bright_cnt, bright_mask, v_thr = green_ratio_and_sat(
            box_hsv, lower_green, upper_green, args.bright_fraction
        )
        if bright_mask is None or bright_cnt == 0:
            continue

        box_bgr = img_bgr[ymin:ymax, xmin:xmax]
        mean_chroma, mean_exg, mean_lab_b = lab_chroma_exg_bright(box_bgr, bright_mask)
        rows.append(
            {
                "score": float(row.get("score", 0.0)),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "green_ratio": green_ratio,
                "mean_sat_bright": mean_sat_bright,
                "mean_chroma": mean_chroma,
                "mean_exg": mean_exg,
                "mean_lab_b": mean_lab_b,
                "v_thr": v_thr,
                "bright_cnt": bright_cnt,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["score"], ascending=False)
    out_csv = base + "_color_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
