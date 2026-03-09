import os
import math
import numpy as np
import pandas as pd

# Ensure we're using an interactive backend (prevents slow/fragile auto-detection).
# Must be set before importing pyplot.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

SATELLITE_IMAGE_PATH = "../dataset/satellite_trees/RaleighSatellite.png"

IMAGE_TOP_LEFT = (35.776006, -78.644325)
IMAGE_BOTTOM_RIGHT = (35.772600, -78.637597)

STAR_START_IMAGE = "images/image_00000303.png"
STAR_END_IMAGE = "images/image_00000590.png"

MAKE_SQUARE_VIEW = True
ZOOM_PAD_FRAC = 0.15
ZOOM_PAD_MIN_M = 5.0

STAR_LABEL = "no-tree zone"

DEBUG_MASK_COUNTS = False  # set True to print how many points fall in the no-tree zone per series


# =========================
# Helpers
# =========================


def get_image_extent_meters(top_left, bottom_right):
    lat1, lon1 = top_left
    lat2, lon2 = bottom_right
    R = 6378137.0
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dlat = math.radians(abs(lat1 - lat2))
    dlon = math.radians(abs(lon1 - lon2))
    height = R * dlat
    width = R * dlon * math.cos(mean_lat)
    return [-width / 2, width / 2, -height / 2, height / 2]


def build_star_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask over df.index marking the no-tree zone (inclusive)."""
    if "image" not in df.columns:
        return pd.Series(False, index=df.index)

    start = df.index[df["image"] == STAR_START_IMAGE]
    end = df.index[df["image"] == STAR_END_IMAGE]

    if len(start) == 0 or len(end) == 0:
        return pd.Series(False, index=df.index)

    lo, hi = sorted([int(start[0]), int(end[0])])
    return (df.index >= lo) & (df.index <= hi)


def _align_mask(mask: pd.Series | np.ndarray, to_index: pd.Index) -> pd.Series:
    """Return a boolean Series aligned to *to_index*.

    This avoids failures when *mask* accidentally becomes a numpy array.
    """
    if isinstance(mask, pd.Series):
        return mask.reindex(to_index).fillna(False).astype(bool)

    # Fall back: treat it as an array aligned to the original df.index.
    # We can't know the original index here, so this is a best-effort that
    # matches by positional index values when possible.
    try:
        return pd.Series(mask, index=to_index, dtype=bool).fillna(False)
    except Exception:
        return pd.Series(False, index=to_index)


def square_zoom(ax, xs_list, ys_list):
    xs = np.concatenate(xs_list) if xs_list else np.array([0.0])
    ys = np.concatenate(ys_list) if ys_list else np.array([0.0])

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    pad = max(span * ZOOM_PAD_FRAC, ZOOM_PAD_MIN_M)
    half = span / 2.0 + pad

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


# =========================
# Main Plot Function
# =========================


def plot_rtk_analysis(csv_path):
    if not os.path.exists(csv_path):
        print("CSV not found.")
        return

    df = pd.read_csv(csv_path)

    rtk_x, rtk_y = "rtk_x", "rtk_y"
    gps_x, gps_y = "gps_x", "gps_y"
    filt_x, filt_y = (
        "manual_ground__manual_sat__rtk_filtered__x",
        "manual_ground__manual_sat__rtk_filtered__y",
    )

    # Compute errors (NaNs propagate)
    df["gps_error"] = np.sqrt((df[gps_x] - df[rtk_x]) ** 2 + (df[gps_y] - df[rtk_y]) ** 2)
    df["filt_error"] = np.sqrt((df[filt_x] - df[rtk_x]) ** 2 + (df[filt_y] - df[rtk_y]) ** 2)

    star_mask = build_star_mask(df)

    # =========================================================
    # Plot 1: Spatial Overlay
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    extent = tuple(get_image_extent_meters(IMAGE_TOP_LEFT, IMAGE_BOTTOM_RIGHT))
    if os.path.exists(SATELLITE_IMAGE_PATH):
        img = plt.imread(SATELLITE_IMAGE_PATH)
        ax1.imshow(img, extent=extent, alpha=0.35, zorder=0)

    def plot_series(xcol, ycol, color, label):
        d = df[[xcol, ycol]].dropna()
        if d.empty:
            return

        # Align star mask to this filtered dataframe robustly
        m = _align_mask(star_mask, d.index)

        # continuous line (no markers)
        ax1.plot(d[xcol], d[ycol], linestyle="-", color=color, linewidth=1.5)

        # circles (normal) - filled circles
        normal = d.loc[~m]
        ax1.plot(
            normal[xcol], normal[ycol],
            linestyle="None",
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )

        # circles (no-tree zone) - hollow circles (outline only)
        zone = d.loc[m]
        ax1.plot(
            zone[xcol], zone[ycol],
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.8,
            color=color,
            label=f"{label} ({STAR_LABEL})",
        )

        if DEBUG_MASK_COUNTS:
            print(f"[plot_series] {label}: total={len(d)} zone={int(m.sum())} normal={int((~m).sum())}")

    plot_series(rtk_x, rtk_y, "red", "RTK (Reference)")
    plot_series(gps_x, gps_y, "blue", "GPS")
    plot_series(filt_x, filt_y, "purple", "TreeLoc")

    # square zoom based on available positions
    xs, ys = [], []
    for xcol, ycol in [(rtk_x, rtk_y), (gps_x, gps_y), (filt_x, filt_y)]:
        d = df[[xcol, ycol]].dropna()
        if not d.empty:
            xs.append(d[xcol].to_numpy())
            ys.append(d[ycol].to_numpy())

    if xs and ys:
        square_zoom(ax1, xs, ys)

    ax1.set_title("Spatial Position vs Satellite Overlay")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3)

    # =========================================================
    # Plot 2: Error Over Time (Continuous Line + Mixed Markers)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    def plot_error(series, color, label):
        # continuous line (draw once)
        ax2.plot(df.index, df[series], linestyle="-", color=color, linewidth=1.8)

        # only place markers where values exist
        valid = df[series].notna()
        idx_valid = df.index[valid]

        # IMPORTANT: star_mask is defined on df.index; don't reindex it to idx_valid
        # in a way that loses alignment. Split by mask, then intersect with valid.
        zone_mask_full = _align_mask(star_mask, df.index)
        idx_star = df.index[valid & zone_mask_full]
        idx_norm = df.index[valid & ~zone_mask_full]

        ax2.plot(
            idx_norm,
            df.loc[idx_norm, series],
            linestyle="None",
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )
        ax2.plot(
            idx_star,
            df.loc[idx_star, series],
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.8,
            color=color,
            label=f"{label} ({STAR_LABEL})",
        )

        if DEBUG_MASK_COUNTS:
            print(f"[plot_error] {label}: valid={int(valid.sum())} zone={len(idx_star)} normal={len(idx_norm)}")

    plot_error("gps_error", "blue", "GPS Error")
    plot_error("filt_error", "purple", "TreeLoc Error")

    ax2.set_title("Error Magnitude Over Time (Relative to RTK)")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Error Distance (m)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.show()


if __name__ == "__main__":
    plot_rtk_analysis("../batch_results_old.csv")