"""Run AutomatedSGIL and plot trajectories over the satellite image.

This is the same idea as `plot_csv.py`, but instead of loading a batch-results
CSV it runs the localization pipeline directly and plots:
  - RTK (reference)
  - GPS (raw)
  - TreeLoc / SGIL estimate

For now we intentionally *do not plot* frames that were skipped due to:
  - IMAGES_TO_SKIP ranges
  - AUTOMATED_SKIP_IF_NO_TREES (no segmentations)

Usage (repo root):
    uv run python -m project_sgil.scripts.plot_run_automated_sgil

Notes
-----
- Coordinates are plotted in the local XY meters frame used by Converter:
    +x = East, +y = North
- The background satellite image is drawn with an approximate meter extent.

"""

from __future__ import annotations

import math
import os

import numpy as np

# Ensure we're using an interactive backend (prevents slow/fragile auto-detection).
# Must be set before importing pyplot.
import matplotlib

# Try to select an interactive backend. Some Python builds on Windows don't ship Tk.
# If none can be loaded, matplotlib will use Agg; in that case we save a PNG.
for _backend in ("TkAgg", "Qt5Agg", "WXAgg", "MacOSX"):  # MacOSX harmless on Windows
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt  # noqa: E402

# Delayed imports so the matplotlib backend is set before pyplot is imported.
from project_sgil.constants import (  # noqa: E402
    IMAGE_BOTTOM_RIGHT,
    IMAGE_TOP_LEFT,
    SATELLITE_IMAGE_PATH,
)
from project_sgil.localization.tree_matcher import TreeMatcher  # noqa: E402
from project_sgil.automated_sgil import AutomatedSGIL  # noqa: E402


def _square_zoom(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, *, zoom: float = 1.0) -> None:
    """Set square axis limits around the provided points.

    Parameters
    ----------
    zoom:
        Multiplier applied to the computed (padded) half-span.
        - 1.0 keeps the current behavior.
        - <1.0 zooms in (shows a smaller area).
        - >1.0 zooms out.

    This leaves the padding logic unchanged; it only scales the final window.
    """
    if xs.size == 0 or ys.size == 0:
        return

    if not np.isfinite(zoom) or zoom <= 0:
        raise ValueError(f"zoom must be a positive finite number, got {zoom!r}")

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    pad = max(0.15 * span, 5.0)
    half = span / 2.0 + pad

    half *= float(zoom)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def get_image_extent_meters(
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return [xmin, xmax, ymin, ymax] for imshow extent in meters.

    This mirrors the approach in plot_csv.py: approximate the bbox size in meters
    using a spherical earth model and place the rect centered at the origin.

    This is close enough for a faint background overlay.
    """

    lat1, lon1 = top_left
    lat2, lon2 = bottom_right
    r = 6378137.0
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dlat = math.radians(abs(lat1 - lat2))
    dlon = math.radians(abs(lon1 - lon2))
    height = r * dlat
    width = r * dlon * math.cos(mean_lat)
    return (-width / 2, width / 2, -height / 2, height / 2)


def run_and_plot() -> None:
    # Run the pipeline
    sgil = AutomatedSGIL()

    # Ensure TreeMatcher is constructed (AutomatedSGIL already does this, but be explicit).
    if not hasattr(sgil, "tree_matcher") or sgil.tree_matcher is None:
        sgil.tree_matcher = TreeMatcher(plot_wedges=False)

    results = sgil.run()

    # We want to plot:
    #   - RTK (always present)
    #   - GPS (present when gps_lat/lon exists)
    #   - TreeLoc / SGIL estimate
    # And render skipped/no-tree frames (matched=False) as hollow circles.

    rtk = [r for r in results if r.rtk_pose is not None]
    gps = [r for r in results if r.gps_pose is not None]
    est = [r for r in results if r.estimated_pose is not None]

    if not rtk:
        print("No RTK points to plot.")
        return

    # =========================
    # Figure 1: Spatial overlay
    # =========================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Satellite background (optional)
    sat_path = SATELLITE_IMAGE_PATH
    extent = get_image_extent_meters(
        (float(IMAGE_TOP_LEFT.x), float(IMAGE_TOP_LEFT.y)),
        (float(IMAGE_BOTTOM_RIGHT.x), float(IMAGE_BOTTOM_RIGHT.y)),
    )

    if os.path.exists(sat_path):
        img = plt.imread(sat_path)
        ax.imshow(img, extent=extent, alpha=0.35, zorder=0)
    else:
        print(f"Satellite image not found: {sat_path!r} (plotting without background)")

    def _plot_pose_series(
        series: list,
        *,
        get_xy,
        color: str,
        label: str,
        show_hollow_skip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plot a pose series as a line + filled markers, with hollow markers for no-tree zone."""
        if not series:
            return np.array([], dtype=float), np.array([], dtype=float)

        xs = np.array([float(get_xy(r)[0]) for r in series], dtype=float)
        ys = np.array([float(get_xy(r)[1]) for r in series], dtype=float)

        # continuous line
        ax.plot(xs, ys, linestyle="-", color=color, linewidth=1.5)

        # markers split by matched
        if show_hollow_skip:
            matched_mask = np.array([bool(getattr(r, "matched", True)) for r in series], dtype=bool)
            xs_m, ys_m = xs[matched_mask], ys[matched_mask]
            xs_s, ys_s = xs[~matched_mask], ys[~matched_mask]

            if xs_m.size:
                ax.plot(
                    xs_m,
                    ys_m,
                    linestyle="None",
                    marker="o",
                    markersize=4,
                    color=color,
                    label=label,
                )
            if xs_s.size:
                ax.plot(
                    xs_s,
                    ys_s,
                    linestyle="None",
                    marker="o",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                    color=color,
                    label=f"{label} (no-tree zone)",
                )
        else:
            ax.plot(
                xs,
                ys,
                linestyle="None",
                marker="o",
                markersize=4,
                color=color,
                label=label,
            )

        return xs, ys

    # Plot RTK (all are "matched" conceptually; render as normal)
    rtk_xs, rtk_ys = _plot_pose_series(
        rtk,
        get_xy=lambda r: (r.rtk_pose.x, r.rtk_pose.y),
        color="red",
        label="RTK",
        show_hollow_skip=False,
    )

    # Plot GPS: hollow when frame was no-tree zone/no-tree so you can see the data gaps
    gps_xs, gps_ys = _plot_pose_series(
        gps,
        get_xy=lambda r: (r.gps_pose.x, r.gps_pose.y),
        color="blue",
        label="GPS",
        show_hollow_skip=True,
    )

    # Plot TreeLoc: for no-tree zone frames estimated_pose is snapped pose;
    # render hollow for no-tree zone
    est_xs, est_ys = _plot_pose_series(
        est,
        get_xy=lambda r: (r.estimated_pose.x, r.estimated_pose.y),
        color="purple",
        label="TreeLoc",
        show_hollow_skip=True,
    )

    all_x = np.concatenate([a for a in (rtk_xs, gps_xs, est_xs) if a.size])
    all_y = np.concatenate([a for a in (rtk_ys, gps_ys, est_ys) if a.size])
    _square_zoom(
        ax,
        all_x,
        all_y,
        zoom=0.85,  # <1.0 zooms in slightly; try 0.9/0.8/0.7
    )

    ax.set_title("RTK vs GPS vs TreeLoc — Satellite Overlay")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # =========================
    # Figure 2: Errors over time
    # =========================
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    def _plot_error(
        *,
        y_values: np.ndarray,
        matched_mask: np.ndarray,
        color: str,
        label: str,
    ) -> None:
        """Plot continuous error line + filled markers.

        Hollow markers denote no-tree zone frames.
        """
        if y_values.size == 0:
            return

        x_idx = np.arange(y_values.size, dtype=float)
        ax2.plot(x_idx, y_values, linestyle="-", color=color, linewidth=1.8)

        # Only put markers where values are finite
        valid_mask = np.isfinite(y_values)
        m = matched_mask & valid_mask
        s = (~matched_mask) & valid_mask

        if np.any(m):
            ax2.plot(
                x_idx[m],
                y_values[m],
                linestyle="None",
                marker="o",
                markersize=4,
                color=color,
                label=label,
            )
        if np.any(s):
            ax2.plot(
                x_idx[s],
                y_values[s],
                linestyle="None",
                marker="o",
                markersize=7,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.8,
                color=color,
                label=f"{label} (no-tree zone)",
            )

    # Use the same frame order as `results`
    matched_mask_full = np.array(
        [bool(getattr(r, "matched", True)) for r in results],
        dtype=bool,
    )

    gps_err = np.array(
        [float(getattr(r, "gps_err_m", float("nan"))) for r in results],
        dtype=float,
    )
    snapped_err = np.array(
        [float(getattr(r, "snapped_err_m", float("nan"))) for r in results],
        dtype=float,
    )
    sgil_err = np.array(
        [float(getattr(r, "sgil_err_m", float("nan"))) for r in results],
        dtype=float,
    )

    _plot_error(y_values=gps_err, matched_mask=matched_mask_full, color="blue", label="GPS Error")
    _plot_error(
        y_values=snapped_err,
        matched_mask=matched_mask_full,
        color="green",
        label="Snapped Error",
    )
    _plot_error(
        y_values=sgil_err,
        matched_mask=matched_mask_full,
        color="purple",
        label="TreeLoc/SGIL Error",
    )

    ax2.set_title("Error Magnitude Over Time (Relative to RTK)")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Error Distance (m)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Show if interactive; otherwise save.
    backend = str(matplotlib.get_backend()).lower()
    is_interactive = not ("agg" in backend)

    if is_interactive:
        plt.show()
    else:
        out_path1 = os.path.join(os.getcwd(), "automated_sgil_overlay.png")
        out_path2 = os.path.join(os.getcwd(), "automated_sgil_errors.png")
        # bbox_inches='tight' can crop right up to titles/legends; add a small
        # pad so there's always visible whitespace on the top/top-right.
        fig.savefig(out_path1, dpi=150, bbox_inches="tight", pad_inches=0.25)
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight", pad_inches=0.25)
        print(
            f"Matplotlib backend '{matplotlib.get_backend()}' is non-interactive; "
            f"saved plots to: {out_path1} and {out_path2}"
        )


if __name__ == "__main__":
    run_and_plot()
