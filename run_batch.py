"""Batch runner for AutomatedSGIL across multiple configurations.

Runs every combination of:
  - ground source  (manual / auto)
  - satellite source (manual / auto)
  - heading field  (rtk_heading_filtered, noisy_1 … noisy_20)

and writes a single CSV where each row is one image frame, with
fixed columns for GPS and RTK ground truth and then per-config
columns for the SGIL estimated x, y, heading.

Usage:
    python run_batch.py                     # default output
    python run_batch.py -o my_results.csv   # custom output path
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration tables — edit these to add / remove runs
# ---------------------------------------------------------------------------


@dataclass
class GroundConfig:
    label: str
    data_logger_path: str


@dataclass
class SatelliteConfig:
    label: str
    tree_locations_path: str


@dataclass
class HeadingConfig:
    label: str
    json_field: str
    heading_error_deg: float
    heading_sweep_range_deg: float


GROUND_CONFIGS: list[GroundConfig] = [
    GroundConfig(
        "manual_ground",
        "dataset/tables/manual_json_data_w_gaussian.json",
    ),
    GroundConfig(
        "auto_ground",
        "dataset/tables/auto_json_data_w_gaussian.json",
    ),
]

SATELLITE_CONFIGS: list[SatelliteConfig] = [
    SatelliteConfig(
        "manual_sat",
        "dataset/tables/RaleighSatellite_manual_trees.csv",
    ),
    SatelliteConfig(
        "auto_sat",
        "dataset/tables/RaleighSatellite_deepforest_trees_latlon.csv",
    ),
]

# Per-heading tuning: adjust HEADING_ERROR_DEG and HEADING_SWEEP_RANGE_DEG
# for each noise level.  The values below are initial estimates — tweak
# as needed.
HEADING_CONFIGS: list[HeadingConfig] = [
    HeadingConfig("rtk_filtered", "rtk_heading_filtered",
                  heading_error_deg=5, heading_sweep_range_deg=4),
    HeadingConfig("noisy_1", "noisy_1_heading",
                  heading_error_deg=7, heading_sweep_range_deg=5),
    HeadingConfig("noisy_5", "noisy_5_heading",
                  heading_error_deg=7, heading_sweep_range_deg=8),
    HeadingConfig("noisy_10", "noisy_10_heading",
                  heading_error_deg=12, heading_sweep_range_deg=12),
    HeadingConfig("noisy_15", "noisy_15_heading",
                  heading_error_deg=17, heading_sweep_range_deg=12),
    HeadingConfig("noisy_20", "noisy_20_heading",
                  heading_error_deg=17, heading_sweep_range_deg=23),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_tag(ground: GroundConfig, sat: SatelliteConfig, heading: HeadingConfig) -> str:
    """Short human-readable tag for one configuration."""
    return f"{ground.label}__{sat.label}__{heading.label}"


def _patch_constants(
    heading_json_field: str,
    heading_error_deg: float,
    heading_sweep_range_deg: float,
    tree_locations_path: str,
) -> None:
    """Monkey-patch the constants module *and* every consumer that used
    ``from constants import X`` (which creates local bindings that won't
    see changes to the constants module alone)."""
    import project_sgil.constants as C

    C.HEADING_JSON_FIELD = heading_json_field
    C.HEADING_ERROR_DEG = heading_error_deg
    C.HEADING_SWEEP_RANGE_DEG = heading_sweep_range_deg
    C.TREE_LOCATIONS_PATH = tree_locations_path

    # Recompute derived constant
    C.AOI_ANGLE_DEG = (C.H_FOV_DEG + C.HEADING_ERROR_DEG) / 2

    # Disable all plotting for speed
    C.PLOT = False
    C.PLOT_WEDGES = False
    C.PLOT_THETAS = False

    # --- Patch local bindings in consuming modules ---
    import project_sgil.automated_sgil as _asgil
    _asgil.HEADING_JSON_FIELD = heading_json_field
    _asgil.PLOT = False
    _asgil.PLOT_WEDGES = False
    _asgil.PLOT_THETAS = False

    import project_sgil.localization.tree_matcher as _tm
    _tm.HEADING_ERROR_DEG = heading_error_deg
    _tm.HEADING_SWEEP_RANGE_DEG = heading_sweep_range_deg
    _tm.AOI_ANGLE_DEG = C.AOI_ANGLE_DEG
    _tm.TREE_LOCATIONS_PATH = tree_locations_path


def _run_single(
    ground: GroundConfig,
    sat: SatelliteConfig,
    heading: HeadingConfig,
) -> dict[str, dict[str, Any]]:
    """Run AutomatedSGIL for one configuration and return a dict keyed
    by image name with the estimated pose + ground truth info.

    Returns:
        { image_name: {
            "rtk_x", "rtk_y", "rtk_heading",
            "gps_x", "gps_y", "gps_heading",
            "est_x", "est_y", "est_heading",
            "matched",
          }, ... }
    """
    # Patch constants before importing / constructing AutomatedSGIL so the
    # module-level reads pick up the right values.
    _patch_constants(
        heading_json_field=heading.json_field,
        heading_error_deg=heading.heading_error_deg,
        heading_sweep_range_deg=heading.heading_sweep_range_deg,
        tree_locations_path=sat.tree_locations_path,
    )

    # Re-import to pick up patched constants.  AutomatedSGIL reads some
    # constants at import-time, but the critical ones (HEADING_JSON_FIELD,
    # HEADING_ERROR_DEG, etc.) are read at call-time via the constants module,
    # so patching above is sufficient.
    from project_sgil.automated_sgil import AutomatedSGIL
    from project_sgil.localization.tree_matcher import TreeMatcher

    sgil = AutomatedSGIL(data_log_path=ground.data_logger_path)
    # TreeMatcher's default path= was bound at import time, so we must
    # explicitly construct with the desired satellite tree CSV.
    sgil.tree_matcher = TreeMatcher(
        plot_wedges=False, path=sat.tree_locations_path,
    )
    results = sgil.run()

    rows: dict[str, dict[str, Any]] = {}
    for r in results:
        row: dict[str, Any] = {}

        # RTK ground truth (always available)
        row["rtk_x"] = r.rtk_pose.x
        row["rtk_y"] = r.rtk_pose.y
        row["rtk_heading"] = r.rtk_pose.yaw

        # GPS
        if r.gps_pose is not None:
            row["gps_x"] = r.gps_pose.x
            row["gps_y"] = r.gps_pose.y
            row["gps_heading"] = r.gps_pose.yaw
        else:
            row["gps_x"] = float("nan")
            row["gps_y"] = float("nan")
            row["gps_heading"] = float("nan")

        # SGIL estimate
        if r.estimated_pose is not None:
            row["est_x"] = r.estimated_pose.x
            row["est_y"] = r.estimated_pose.y
            row["est_heading"] = r.estimated_pose.yaw
        else:
            row["est_x"] = float("nan")
            row["est_y"] = float("nan")
            row["est_heading"] = float("nan")

        row["matched"] = r.matched
        rows[r.image_name] = row

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SGIL runner")
    parser.add_argument(
        "-o", "--output", default="batch_results.csv",
        help="Path for the output CSV (default: batch_results.csv)",
    )
    args = parser.parse_args()

    # Build the list of all (ground, sat, heading) combos
    combos: list[tuple[GroundConfig, SatelliteConfig, HeadingConfig]] = []
    for g in GROUND_CONFIGS:
        for s in SATELLITE_CONFIGS:
            for h in HEADING_CONFIGS:
                combos.append((g, s, h))

    total = len(combos)
    print(f"=== Batch runner: {total} configurations ===\n")

    # Run each config, collecting {image_name: row_dict} per run.
    all_run_results: list[tuple[str, dict[str, dict[str, Any]]]] = []

    for i, (g, s, h) in enumerate(combos, 1):
        tag = _run_tag(g, s, h)
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {tag}")
        print(f"  ground  = {g.data_logger_path}")
        print(f"  sat     = {s.tree_locations_path}")
        print(f"  heading = {h.json_field}  "
              f"(error={h.heading_error_deg}°, sweep=±{h.heading_sweep_range_deg}°)")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            run_rows = _run_single(g, s, h)
        except Exception as exc:
            print(f"  *** FAILED: {exc}")
            run_rows = {}
        elapsed = time.perf_counter() - t0
        print(f"  -> {len(run_rows)} frames in {elapsed:.1f}s")

        all_run_results.append((tag, run_rows))

    # ------------------------------------------------------------------
    # Merge into a single DataFrame
    # ------------------------------------------------------------------
    # Collect the union of all image names across all runs.
    all_images: set[str] = set()
    for _, rr in all_run_results:
        all_images.update(rr.keys())

    # Sort images by numeric index extracted from the name.
    def _sort_key(name: str) -> int:
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    sorted_images = sorted(all_images, key=_sort_key)

    # Build the output rows.
    # Fixed columns come from the first run that has data for each image.
    output_rows: list[dict[str, Any]] = []

    for img in sorted_images:
        row: dict[str, Any] = {"image": img}

        # Grab ground truth from whichever run has this image.
        gt_filled = False
        for _, rr in all_run_results:
            if img in rr and not gt_filled:
                row["gps_x"] = rr[img]["gps_x"]
                row["gps_y"] = rr[img]["gps_y"]
                row["gps_heading"] = rr[img]["gps_heading"]
                row["rtk_x"] = rr[img]["rtk_x"]
                row["rtk_y"] = rr[img]["rtk_y"]
                row["rtk_heading"] = rr[img]["rtk_heading"]
                gt_filled = True
                break

        if not gt_filled:
            row["gps_x"] = row["gps_y"] = row["gps_heading"] = float("nan")
            row["rtk_x"] = row["rtk_y"] = row["rtk_heading"] = float("nan")

        # Per-config columns
        for tag, rr in all_run_results:
            if img in rr:
                row[f"{tag}__x"] = rr[img]["est_x"]
                row[f"{tag}__y"] = rr[img]["est_y"]
                row[f"{tag}__heading"] = rr[img]["est_heading"]
            else:
                row[f"{tag}__x"] = float("nan")
                row[f"{tag}__y"] = float("nan")
                row[f"{tag}__heading"] = float("nan")

        output_rows.append(row)

    df = pd.DataFrame(output_rows)
    df.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(df)} rows × {len(df.columns)} columns to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
