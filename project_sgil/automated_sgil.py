"""Automated SGIL: localization from segmentation centroids in results.json.

Unlike `manual_sgil` (which requires clicking tree centers), this script uses
pre-computed tree detections stored per-frame in `dataset/tables/results.json`.

It filters which frames to process using constants in `project_sgil.constants`.

author: Jack Elia
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from typing import Any

import pandas as pd

from project_sgil.constants import (
    AUTOMATED_FRAME_END,
    AUTOMATED_FRAME_START,
    AUTOMATED_MIN_SEGMENT_CONFIDENCE,
    AUTOMATED_SKIP_IF_NO_TREES,
    DATA_LOGGER_PATH,
    HEADING_JSON_FIELD,
    HEADING_SWEEP_ENABLED,
    IMAGE_SHAPE,
    IMAGES_TO_SKIP,
    INCLUDE_SKIPPED_FRAMES,
    ORIGIN,
    PLOT,
    PLOT_WEDGES,
    PLOT_THETAS,
    RTK_TO_CAMERA_OFFSET_X_FWD_M,
    RTK_TO_CAMERA_OFFSET_Y_LEFT_M,
)
from project_sgil.data_structs import LocalizationResult, Point, Pose2d
from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter
from project_sgil.localization.road_matcher import RoadMatcher


class AutomatedSGIL:
    """Runs SGIL matching using segmentation centroids from results.json."""

    def __init__(
        self,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
        specific_indices: list[int] | None = None,
        start_index: int = AUTOMATED_FRAME_START,
        only_on_new_gps: bool = True,
        latlon_epsilon: float = 1e-9,
    ) -> None:
        self.converter = Converter(origin[0], origin[1])
        self.tree_matcher = TreeMatcher(PLOT_WEDGES)
        self.road_matcher = RoadMatcher(self.converter)
        self.image_shape = IMAGE_SHAPE

        # If provided, process exactly these numeric frame indices (e.g. [140, 152, ...]).
        # Otherwise fall back to AUTOMATED_FRAME_START/END/STEP filtering.
        self.specific_indices: set[int] | None = (
            {int(i) for i in specific_indices} if specific_indices else None
        )

        # Automatic selection: start at start_index, then (optionally) only
        # process frames when GPS/RTK lat/lon changes (skips intermediate images
        # between pose updates).
        self.start_index = int(start_index)
        self.only_on_new_gps = bool(only_on_new_gps)
        self.latlon_epsilon = float(latlon_epsilon)
        self._last_selected_latlon: tuple[float, float] | None = None

        # Load JSON frames once
        self.data_log_path = self._resolve_json_path(data_log_path)
        self.frame_lookup = self._load_frame_lookup(self.data_log_path)

        # Runtime state
        self.current_pose: Pose2d | None = None
        self._last_rtk_xy: tuple[float, float] | None = None

        self._correct_pose_latlon: tuple[float, float] | None = None
        self._rtk_pose_latlon: tuple[float, float] | None = None
        self._gps_pose_latlon: tuple[float, float] | None = None

    @staticmethod
    def _frame_latlon(frame: dict[str, Any]) -> tuple[float, float] | None:
        """Return the best available (lat, lon) tuple for detecting new pose updates."""
        try:
            # IMPORTANT: use GPS first. RTK often changes every frame, but GPS updates
            # slower (e.g., ~2 Hz). We want to skip images between GPS updates.
            lat = frame.get("gps_lat", None)
            lon = frame.get("gps_lon", None)
            if lat is None or lon is None:
                lat = frame.get("rtk_lat", None)
                lon = frame.get("rtk_lon", None)
            if lat is None or lon is None:
                return None
            return (float(lat), float(lon))
        except (TypeError, ValueError):
            return None

    def _latlon_changed(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> bool:
        """Return True if (lat, lon) pairs differ by more than epsilon."""
        return (
            abs(a[0] - b[0]) > self.latlon_epsilon
            or abs(a[1] - b[1]) > self.latlon_epsilon
        )

    def _passes_new_gps_gate(self, frame: dict[str, Any]) -> bool:
        if not self.only_on_new_gps:
            return True
        latlon = self._frame_latlon(frame)
        if latlon is None:
            # If we can't read a pose, don't gate on it.
            return True
        if self._last_selected_latlon is None:
            self._last_selected_latlon = latlon
            return True
        if self._latlon_changed(latlon, self._last_selected_latlon):
            self._last_selected_latlon = latlon
            return True
        return False

    # ---------------- JSON helpers (mirrors manual_sgil) ----------------

    @staticmethod
    def _resolve_json_path(path: str) -> str:
        if path.lower().endswith(".json"):
            return path
        if path.lower().endswith(".csv"):
            return path[:-4] + ".json"
        return path + ".json"

    @staticmethod
    def _load_frame_lookup(path: str) -> dict[str, dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON log not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        lookup: dict[str, dict[str, Any]] = {}
        for frame in frames:
            name = str(frame.get("frame_name", ""))
            if not name:
                continue
            lookup[name] = frame
            lookup[os.path.basename(name)] = frame
        return lookup

    def _get_frame_for_image(self, image_name: str) -> dict[str, Any] | None:
        frame = self.frame_lookup.get(image_name)
        if frame is not None:
            return frame

        base = os.path.basename(image_name)
        frame = self.frame_lookup.get(base)
        if frame is not None:
            return frame

        # try extension swap
        root, _ = os.path.splitext(base)
        for ext in (".jpg", ".jpeg", ".png"):
            cand = f"{root}{ext}"
            frame = self.frame_lookup.get(cand)
            if frame is not None:
                return frame
        return None

    # ---------------- Frame filtering ----------------

    @staticmethod
    def _frame_index_from_name(frame_name: str) -> int | None:
        """Extract numeric index from names like images/image_00000140.png."""
        base = os.path.basename(frame_name)
        m = re.search(r"(\d+)", base)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def _frame_selected(self, idx: int) -> bool:
        """Return True if this numeric frame index should be processed."""
        if self.specific_indices is not None:
            return idx in self.specific_indices

        # New default: start at start_index, then let GPS gating decide.
        if idx < self.start_index:
            return False

        # Keep end bound if configured
        if AUTOMATED_FRAME_END is not None and idx > AUTOMATED_FRAME_END:
            return False

        return True

    # ---------------- Pose + trees ----------------

    def _frame_to_pose(self, frame: dict[str, Any]) -> Pose2d | None:
        if frame.get(HEADING_JSON_FIELD) is None:
            return None

        # Store lat/lon (raw antenna locations)
        self._rtk_pose_latlon = (float(frame["rtk_lat"]), float(frame["rtk_lon"]))
        self._gps_pose_latlon = (float(frame["gps_lat"]), float(frame["gps_lon"]))

        # RTK antenna position in XY
        x, y = self.converter.latlon_to_xy(self._rtk_pose_latlon)

        # Yaw from the configured heading field
        yaw = self.converter.rtk_heading_to_yaw(frame[HEADING_JSON_FIELD])

        # Apply extrinsic: RTK -> camera (body frame offsets rotated by yaw)
        dx_b = float(RTK_TO_CAMERA_OFFSET_X_FWD_M)
        dy_b = float(RTK_TO_CAMERA_OFFSET_Y_LEFT_M)
        c = math.cos(math.radians(yaw))
        s = math.sin(math.radians(yaw))
        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        # RTK camera position
        x = x - dx_w
        y = y - dy_w

        # Also compute GPS camera-position XY for delta propagation when desired
        gps_x, gps_y = self.converter.latlon_to_xy(self._gps_pose_latlon)
        gps_x = gps_x - dx_w
        gps_y = gps_y - dy_w
        self._gps_xy = (gps_x, gps_y)

        self._correct_pose_latlon = self._rtk_pose_latlon
        return Pose2d(x=x, y=y, yaw=yaw)

    def _pixel_points_from_segmentations(self, frame: dict[str, Any]) -> list[Point]:
        segs = frame.get("segmentations") or []
        points: list[Point] = []
        for seg in segs:
            try:
                conf = float(seg.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < AUTOMATED_MIN_SEGMENT_CONFIDENCE:
                continue
            try:
                cx = float(seg.get("centroid_x"))
                cy = float(seg.get("centroid_y"))
            except (TypeError, ValueError):
                continue
            points.append(Point(cx, cy))
        return points

    def _sgil_error_meters(self, estimated_pose: Pose2d, rtk_pose: Pose2d) -> float:
        """Euclidean error in the local XY frame (meters)."""
        return float(
            math.hypot(
                estimated_pose.x - rtk_pose.x,
                estimated_pose.y - rtk_pose.y,
            )
        )

    @staticmethod
    def _in_skip_range(idx: int) -> bool:
        for a, b in IMAGES_TO_SKIP:
            if a <= idx < b:
                return True
        return False

    def run(self) -> list[LocalizationResult]:
        results: list[LocalizationResult] = []
        if PLOT:
            DebugVisualizer.clear_plots()

        # Iterate over frames sorted by index
        frames: list[dict[str, Any]] = list(self.frame_lookup.values())
        # de-dupe because we store both full path and basename
        unique_frames: dict[str, dict[str, Any]] = {}
        for fr in frames:
            name = str(fr.get("frame_name", ""))
            if name:
                unique_frames[name] = fr

        ordered: list[dict[str, Any]] = sorted(
            unique_frames.values(),
            key=lambda fr: self._frame_index_from_name(str(fr.get("frame_name", "")))
            or 10**18,
        )

        for frame in ordered:
            frame_name = str(frame.get("frame_name", "")).strip()
            idx = self._frame_index_from_name(frame_name)
            if idx is None or not self._frame_selected(idx):
                continue

            # Skip if this frame doesn't correspond to a new GPS/RTK update.
            if not self._passes_new_gps_gate(frame):
                continue

            # Parse pose (also stores gps/rtk latlon)
            rtk_pose = self._frame_to_pose(frame)
            if rtk_pose is None:
                continue

            gps_latlon = self._gps_pose_latlon
            if gps_latlon is None:
                continue

            # Raw GPS camera XY (for reporting only)
            gps_x, gps_y = self.converter.latlon_to_xy(gps_latlon)
            gps_pose = Pose2d(gps_x, gps_y, yaw=rtk_pose.yaw)

            # Snap GPS position to the road and choose road direction closest to GPS yaw.
            road_match = self.road_matcher.match_gps(
                gps_latlon=gps_latlon,
                gps_yaw_deg=rtk_pose.yaw,
            )
            snapped_pose = road_match.snapped_pose

            # Skip ranges: do not run tree matching, but still report a snapped pose.
            if idx is not None and self._in_skip_range(idx):
                self.current_pose = snapped_pose
                if INCLUDE_SKIPPED_FRAMES:
                    results.append(
                        LocalizationResult(
                            image_name=frame_name,
                            sgil_err_m=self._sgil_error_meters(snapped_pose, rtk_pose),
                            snapped_err_m=self._sgil_error_meters(snapped_pose, rtk_pose),
                            gps_err_m=float(
                                math.hypot(gps_x - rtk_pose.x, gps_y - rtk_pose.y)
                            ),
                            rtk_pose=rtk_pose,
                            current_pose=snapped_pose,
                            gps_pose=gps_pose,
                            estimated_pose=snapped_pose,
                            matched=False,
                        )
                    )
                continue

            pixel_points = self._pixel_points_from_segmentations(frame)
            if AUTOMATED_SKIP_IF_NO_TREES and not pixel_points:
                # No trees: skip matching and just use the road-snapped pose.
                self.current_pose = snapped_pose

                # Optionally record this frame (matched=False)
                if INCLUDE_SKIPPED_FRAMES:
                    results.append(
                        LocalizationResult(
                            image_name=frame_name,
                            sgil_err_m=self._sgil_error_meters(snapped_pose, rtk_pose),
                            snapped_err_m=self._sgil_error_meters(snapped_pose, rtk_pose),
                            gps_err_m=float(
                                math.hypot(gps_x - rtk_pose.x, gps_y - rtk_pose.y)
                            ),
                            rtk_pose=rtk_pose,
                            current_pose=snapped_pose,
                            gps_pose=gps_pose,
                            # Best available estimate when no trees are visible.
                            estimated_pose=snapped_pose,
                            matched=False,
                        )
                    )
                continue

            # Convert pixel points -> ground thetas
            img_w = int(self.image_shape[1])
            ground_thetas = [
                self.converter.image_x_to_theta(pt.x, img_w) for pt in pixel_points
            ]

            # Allow per-frame heading override/offset (useful for debugging).
            # Initial yaw seed comes from the road direction. TreeMatcher can
            # refine via heading sweep.
            yaw_for_match = snapped_pose.yaw

            pose_for_match = Pose2d(snapped_pose.x, snapped_pose.y, yaw_for_match)

            try:
                est_pose_from_matcher = self.tree_matcher.match_trees(
                    pose_for_match,
                    ground_thetas,
                    rtk_pose,
                    image_name=frame_name,
                )
                est_pose: Pose2d | None = Pose2d(
                    est_pose_from_matcher.x,
                    est_pose_from_matcher.y,
                    est_pose_from_matcher.yaw,
                )
                matched = True
            except Exception:
                matched = False
                est_pose = None

            # If matching fails, fall back to the road-snapped pose rather than None.
            if not matched or est_pose is None:
                est_pose = snapped_pose

            # Snapshot AOI trees immediately after matching so plotting uses
            # the AOI computed for this exact pose.
            aoi_trees_for_plot = list(getattr(self.tree_matcher, "aoi_trees", []))

            # If heading sweep is disabled, keep using the initial yaw when we have an estimate.
            if est_pose is not None and not HEADING_SWEEP_ENABLED:
                est_pose.yaw = pose_for_match.yaw

            # Advance: keep current_pose aligned with the best available estimate.
            # If heading sweep ran, est_pose.yaw differs from pose_for_match.yaw.
            self.current_pose = est_pose if est_pose is not None else pose_for_match
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            sgil_err_m = (
                self._sgil_error_meters(est_pose, rtk_pose)
                if est_pose is not None
                else float("nan")
            )

            if PLOT and idx:
                save_stub = f"frame_{idx:06d}"

                if PLOT_THETAS:
                    DebugVisualizer.plot_thetas(
                        ground_thetas,
                        pose_for_match,
                        aoi_trees_for_plot,
                        f"{save_stub}_thetas",
                        rtk_pose=rtk_pose,
                    )

                DebugVisualizer.plot_aoi(
                    self.tree_matcher.satellite_tree_locations,
                    aoi_trees_for_plot,
                    pose_for_match,
                    save_stub,
                )

            results.append(
                LocalizationResult(
                    image_name=frame_name,
                    sgil_err_m=sgil_err_m,
                    snapped_err_m=self._sgil_error_meters(snapped_pose, rtk_pose),
                    gps_err_m=float(math.hypot(gps_x - rtk_pose.x, gps_y - rtk_pose.y)),
                    rtk_pose=rtk_pose,
                    current_pose=self.current_pose,
                    gps_pose=gps_pose,
                    estimated_pose=est_pose,
                    matched=matched,
                )
            )

        if results:
            valid_errors = [r.sgil_err_m for r in results if pd.notna(r.sgil_err_m)]

            if valid_errors:
                mean_err = sum(valid_errors) / len(valid_errors)
                median_err = statistics.median(valid_errors)
                print(f"\nMean SGIL error  over {len(valid_errors)} frames: {mean_err:.3f} m")
                print(f"Median SGIL error over {len(valid_errors)} frames: {median_err:.3f} m")

        self._print_summary_table(results)
        return results

    def _print_summary_table(self, results: list[LocalizationResult]) -> None:
        if not results:
            print("No frames processed.")
            return

        def fmt_pose(p: Pose2d) -> str:
            return f"(x={p.x:.2f}, y={p.y:.2f}, yaw={p.yaw:.2f}°)"

        # Helper: safe mean over finite values
        def mean_finite(vals: list[float]) -> float:
            finite = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
            return float(sum(finite) / len(finite)) if finite else float("nan")

        print(
            f"\n{'frame':40s} | {'matched':10s} | {'sgil_err_m':10s} | {'snapped_err_m':13s} | {'gps_err_m':10s} | "
            f"{'gps_pose':32s} | {'rtk_pose':32s} | {'current_pose':32s} | {'estimated_pose':32s}"
        )
        print(
            "-" * 40
            + "-+-"
            + "-" * 10
            + "-+-"
            + "-" * 13
            + "-+-"
            + "-" * 10
            + "-+-"
            + "-" * 32
            + "-+-"
            + "-" * 32
            + "-+-"
            + "-" * 32
            + "-+-"
            + "-" * 32
        )

        # Track which frames were skipped (either via IMAGES_TO_SKIP or due to no trees)
        skipped_names: set[str] = set()
        for r in results:
            # idx-based skip
            idx = self._frame_index_from_name(r.image_name)
            if idx is not None and self._in_skip_range(idx):
                skipped_names.add(r.image_name)
            # no-tree skip path is also matched=False but not in IMAGES_TO_SKIP
            if AUTOMATED_SKIP_IF_NO_TREES and (not r.matched):
                # Mark as skipped when estimated_pose == snapped_pose and we didn't attempt matching.
                # (Conservative: avoids counting failed matches as "skipped".)
                if r.estimated_pose is not None and r.current_pose is not None:
                    if (
                        abs(r.estimated_pose.x - r.current_pose.x) < 1e-9
                        and abs(r.estimated_pose.y - r.current_pose.y) < 1e-9
                        and abs(r.estimated_pose.yaw - r.current_pose.yaw) < 1e-9
                    ):
                        skipped_names.add(r.image_name)

        for r in results:
            print(
                f"{r.image_name:40s} | "
                f"{r.matched!s:10s} | "
                f"{r.sgil_err_m:10.2f} | "
                f"{r.snapped_err_m:13.2f} | "
                f"{r.gps_err_m:10.2f} | "
                f"{fmt_pose(r.gps_pose) if r.gps_pose else 'None':32s} | "
                f"{fmt_pose(r.rtk_pose):32s} | "
                f"{fmt_pose(r.current_pose):32s} | "
                f"{fmt_pose(r.estimated_pose) if r.estimated_pose else 'None':32s}"
            )

        # ---- Summary means (3 columns), with and without skipped images ----
        all_sgil = [r.sgil_err_m for r in results]
        all_snap = [r.snapped_err_m for r in results]
        all_gps = [r.gps_err_m for r in results]

        not_skipped = [r for r in results if r.image_name not in skipped_names]
        ns_sgil = [r.sgil_err_m for r in not_skipped]
        ns_snap = [r.snapped_err_m for r in not_skipped]
        ns_gps = [r.gps_err_m for r in not_skipped]

        print("\n--- Mean errors ---")
        print(
            f"Including skipped (n={len(results)}): "
            f"mean_sgil_err_m={mean_finite(all_sgil):.3f} | "
            f"mean_snapped_err_m={mean_finite(all_snap):.3f} | "
            f"mean_gps_err_m={mean_finite(all_gps):.3f}"
        )
        print(
            f"Excluding skipped (n={len(not_skipped)}): "
            f"mean_sgil_err_m={mean_finite(ns_sgil):.3f} | "
            f"mean_snapped_err_m={mean_finite(ns_snap):.3f} | "
            f"mean_gps_err_m={mean_finite(ns_gps):.3f}"
        )


if __name__ == "__main__":
    AutomatedSGIL().run()
