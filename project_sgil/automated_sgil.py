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
    AUTOMATED_USE_RTK_POSE_EACH_FRAME,
    DATA_LOGGER_PATH,
    HEADING_SWEEP_ENABLED,
    IMAGE_SHAPE,
    IMAGES_TO_SKIP,
    ORIGIN,
    PLOT,
    PLOT_WEDGES,
    PLOT_THETAS,
    MIN_DX,
    MIN_DY,
    RTK_TO_CAMERA_OFFSET_X_FWD_M,
    RTK_TO_CAMERA_OFFSET_Y_LEFT_M,
    HEADING_ERROR_AFTER_SKIP_DEG,
    HEADING_ERROR_AFTER_SUCCESS_DEG,
    H_FOV_DEG,
)
from project_sgil.data_structs import LocalizationResult, Point, Pose2d
from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter


class AutomatedSGIL:
    """Runs SGIL matching using segmentation centroids from results.json."""
# Example: {605: -5.0} means "use the normal yaw - 5 degrees".
    HEADING_OFFSET_DEG: dict[int, float] = {
        605: 0.0,
    }

    # Backwards-compatible alias (deprecated): absolute yaw override.
    # Prefer HEADING_OFFSET_DEG.
    HEADING_OVERRIDE_DEG: dict[int, float] = {}

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
        self._last_gps_xy: tuple[float, float] | None = None
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
            name = str(frame.get("frame_name", "")).strip()
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
        # Require RTK fields for position.
        if frame.get("rtk_heading") is None:
            return None

        # Store lat/lon (raw antenna locations)
        self._rtk_pose_latlon = (float(frame["rtk_lat"]), float(frame["rtk_lon"]))
        self._gps_pose_latlon = (float(frame["gps_lat"]), float(frame["gps_lon"]))

        # RTK antenna position in XY
        x, y = self.converter.latlon_to_xy(self._rtk_pose_latlon)

        # --- Always compute RTK yaw for the RTK pose column ---
        rtk_yaw = self.converter.rtk_heading_to_yaw(float(frame["rtk_heading"]))

        # --- Heading source for matching/current_pose (can differ from RTK yaw) ---
        # If AUTOMATED_USE_RTK_POSE_EACH_FRAME is True, use RTK heading.
        # If False, use gps_heading_filtered after the first processed frame (seed with RTK).
        if AUTOMATED_USE_RTK_POSE_EACH_FRAME:
            match_yaw = rtk_yaw
        else:
            if self.current_pose is None:
                match_yaw = rtk_yaw
            else:
                if frame.get("gps_heading_filtered") is None:
                    match_yaw = rtk_yaw
                else:
                    match_yaw = self.converter.rtk_heading_to_yaw(
                        float(frame["gps_heading_filtered"])
                    )

        # Apply extrinsic: RTK -> camera (body frame offsets rotated by *match* yaw)
        # so camera XY corresponds to whatever yaw you're using for matching.
        dx_b = float(RTK_TO_CAMERA_OFFSET_X_FWD_M)
        dy_b = float(RTK_TO_CAMERA_OFFSET_Y_LEFT_M)
        c = math.cos(math.radians(match_yaw))
        s = math.sin(math.radians(match_yaw))
        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        # Camera position (using RTK lat/lon + extrinsic rotated by match_yaw)
        x_cam = x - dx_w
        y_cam = y - dy_w

        # Also compute GPS camera-position XY for delta propagation when desired
        gps_x, gps_y = self.converter.latlon_to_xy(self._gps_pose_latlon)
        gps_x = gps_x - dx_w
        gps_y = gps_y - dy_w
        self._gps_xy = (gps_x, gps_y)

        self._correct_pose_latlon = self._rtk_pose_latlon

        # Store BOTH yaws so run() can populate the right columns.
        self._rtk_yaw = float(rtk_yaw)
        self._match_yaw = float(match_yaw)

        return Pose2d(x=float(x_cam), y=float(y_cam), yaw=float(match_yaw))

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

    def _set_heading_error_deg(self, deg: float) -> None:
        """Update TreeMatcher's runtime heading tolerance.

        TreeMatcher imports HEADING_ERROR_DEG and AOI_ANGLE_DEG as module-level
        constants, so changing project_sgil.constants at runtime isn't enough.
        We patch the values in the imported tree_matcher module.
        """
        try:
            import project_sgil.localization.tree_matcher as tm

            tm.HEADING_ERROR_DEG = float(deg)
            tm.AOI_ANGLE_DEG = (float(H_FOV_DEG) + float(tm.HEADING_ERROR_DEG)) / 2.0
            print(
                f"[AutomatedSGIL] Set heading error tolerance to {tm.HEADING_ERROR_DEG:.1f} deg"
            )
        except Exception as e:
            print(f"[AutomatedSGIL] Failed to update heading error tolerance: {e}")

    def run(self) -> list[LocalizationResult]:
        results: list[LocalizationResult] = []
        if PLOT:
            DebugVisualizer.clear_plots()

        # Start with the 'normal' tolerance
        self._set_heading_error_deg(float(HEADING_ERROR_AFTER_SUCCESS_DEG))

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

        # When using carryover mode (not RTK-each-frame), we need to continue
        # updating the last GPS anchor across skipped frames so the next processed
        # frame's dx_gps/dy_gps includes the skipped motion.
        for frame in ordered:
            frame_name = str(frame.get("frame_name", "")).strip()
            idx = self._frame_index_from_name(frame_name)
            if idx is None or not self._frame_selected(idx):
                continue

            # Skip if this frame doesn't correspond to a new GPS/RTK update.
            if not self._passes_new_gps_gate(frame):
                continue

            # Parse pose (this also computes current GPS camera XY as self._gps_xy)
            rtk_pose = self._frame_to_pose(frame)
            if rtk_pose is None:
                continue

            # rtk_pose returned above uses the yaw selected for matching; for the
            # table we also want the RTK yaw to be preserved.
            rtk_yaw = float(getattr(self, "_rtk_yaw", rtk_pose.yaw))
            match_yaw = float(getattr(self, "_match_yaw", rtk_pose.yaw))

            rtk_pose_for_table = Pose2d(rtk_pose.x, rtk_pose.y, rtk_yaw)

            # Compute current GPS camera XY (set in _frame_to_pose)
            curr_gps_xy = getattr(self, "_gps_xy", None)

            # If this index is inside a user-configured skip range, don't create
            # a result row (and don't run the matcher). But in carryover mode,
            # keep accumulating GPS deltas by advancing the anchor.
            if idx is not None and self._in_skip_range(idx):
                # We are skipping frames in this range; widen tolerance so the next
                # processed frame after the skip is easier to match.
                self._set_heading_error_deg(float(HEADING_ERROR_AFTER_SKIP_DEG))

                if not AUTOMATED_USE_RTK_POSE_EACH_FRAME and curr_gps_xy is not None:
                    # Initialize anchors/state so we can accumulate deltas across the skipped range
                    if self.current_pose is None:
                        self.current_pose = Pose2d(rtk_pose.x, rtk_pose.y, rtk_pose.yaw)
                    if self._last_gps_xy is None:
                        self._last_gps_xy = (float(curr_gps_xy[0]), float(curr_gps_xy[1]))
                    else:
                        dx_skip = float(curr_gps_xy[0]) - float(self._last_gps_xy[0])
                        dy_skip = float(curr_gps_xy[1]) - float(self._last_gps_xy[1])
                        # Advance the predicted pose by the skipped motion so the
                        # next processed frame starts at the correct location.
                        self.current_pose = Pose2d(
                            self.current_pose.x + dx_skip,
                            self.current_pose.y + dy_skip,
                            self.current_pose.yaw,
                        )
                        self._last_gps_xy = (float(curr_gps_xy[0]), float(curr_gps_xy[1]))
                self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)
                continue

            # Debug: print RTK/GPS positions and deltas each processed step
            if self._last_rtk_xy is None:
                dx_rtk = dy_rtk = 0.0
            else:
                (
                    last_rtk_x,
                    last_rtk_y,
                ) = self._last_rtk_xy
                dx_rtk = rtk_pose.x - last_rtk_x
                dy_rtk = rtk_pose.y - last_rtk_y

            if curr_gps_xy is None or self._last_gps_xy is None:
                dx_gps_dbg = dy_gps_dbg = 0.0
            else:
                dx_gps_dbg = float(curr_gps_xy[0]) - float(self._last_gps_xy[0])
                dy_gps_dbg = float(curr_gps_xy[1]) - float(self._last_gps_xy[1])

            pixel_points = self._pixel_points_from_segmentations(frame)
            if AUTOMATED_SKIP_IF_NO_TREES and not pixel_points:
                continue

            # Initialize state
            if self.current_pose is None:
                self.current_pose = Pose2d(rtk_pose.x, rtk_pose.y, rtk_pose.yaw)
                self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)
                if hasattr(self, "_gps_xy") and self._gps_xy is not None:
                    self._last_gps_xy = (float(self._gps_xy[0]), float(self._gps_xy[1]))

            assert self.current_pose is not None

            # GPS delta since last processed (camera-frame XY)
            curr_gps_xy = getattr(self, "_gps_xy", None)
            if curr_gps_xy is None or self._last_gps_xy is None:
                dx_gps = dy_gps = 0.0
            else:
                dx_gps = float(curr_gps_xy[0]) - float(self._last_gps_xy[0])
                dy_gps = float(curr_gps_xy[1]) - float(self._last_gps_xy[1])

            if AUTOMATED_USE_RTK_POSE_EACH_FRAME:
                # No carryover: use RTK pose directly for matching.
                predicted_x = rtk_pose.x
                predicted_y = rtk_pose.y
            else:
                predicted_x = self.current_pose.x + dx_gps
                predicted_y = self.current_pose.y + dy_gps

            # Convert pixel points -> ground thetas
            img_w = int(self.image_shape[1])
            ground_thetas = [
                self.converter.image_x_to_theta(pt.x, img_w) for pt in pixel_points
            ]

            # Allow per-frame heading override/offset (useful for debugging).
            # Priority:
            #   1) per-frame offset: yaw = rtk_yaw + offset
            #   2) legacy absolute override: yaw = override
            #   3) default: rtk_yaw
            yaw_for_match = match_yaw
            if idx is not None and idx in self.HEADING_OFFSET_DEG:
                yaw_for_match = float(match_yaw) + float(self.HEADING_OFFSET_DEG[idx])
            elif idx is not None and idx in self.HEADING_OVERRIDE_DEG:
                yaw_for_match = float(self.HEADING_OVERRIDE_DEG[idx])

            pose_for_match = Pose2d(predicted_x, predicted_y, yaw_for_match)

            est_pose: Pose2d | None
            if dx_gps > MIN_DX or dy_gps > MIN_DY:
                try:
                    est_xy = self.tree_matcher.match_trees(
                        pose_for_match,
                        ground_thetas,
                        rtk_pose,
                        image_name=frame_name,
                    )
                    est_pose = Pose2d(est_xy.x, est_xy.y, pose_for_match.yaw)
                    matched = True

                    # After a successful localization, restore the normal heading tolerance.
                    self._set_heading_error_deg(float(HEADING_ERROR_AFTER_SUCCESS_DEG))

                except Exception:
                    matched = False
                    est_pose = None
            else:
                # Not enough motion since last GPS update: we skip matching.
                matched = False
                est_pose = None

            # Snapshot AOI trees immediately after matching so plotting uses
            # the AOI computed for this exact pose.
            aoi_trees_for_plot = list(getattr(self.tree_matcher, "aoi_trees", []))

            # If heading sweep is disabled, keep using RTK yaw when we have an estimate.
            if est_pose is not None and not HEADING_SWEEP_ENABLED:
                est_pose.yaw = pose_for_match.yaw

            # Advance
            if AUTOMATED_USE_RTK_POSE_EACH_FRAME:
                # Keep the internal state in sync with RTK if we're in pure-RTK mode.
                self.current_pose = Pose2d(rtk_pose.x, rtk_pose.y, rtk_pose.yaw)
            else:
                # Only advance with SGIL estimate if we actually got one.
                if est_pose is not None:
                    self.current_pose = Pose2d(est_pose.x, est_pose.y, rtk_pose.yaw)
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)
            curr_gps_xy = getattr(self, "_gps_xy", None)
            if curr_gps_xy is not None:
                self._last_gps_xy = (float(curr_gps_xy[0]), float(curr_gps_xy[1]))

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

            # Build a GPS pose record for the table. Use gps_heading_filtered when
            # available so gps_pose doesn't misleadingly inherit RTK yaw.
            gps_pose: Pose2d | None
            if curr_gps_xy is None:
                gps_pose = None
            else:
                gps_yaw = None
                try:
                    if frame.get("gps_heading_filtered") is not None:
                        gps_yaw = self.converter.rtk_heading_to_yaw(
                            float(frame["gps_heading_filtered"])
                        )
                except Exception:
                    gps_yaw = None
                if gps_yaw is None:
                    gps_yaw = float(rtk_pose.yaw)
                gps_pose = Pose2d(float(curr_gps_xy[0]), float(curr_gps_xy[1]), yaw=gps_yaw)

            results.append(
                LocalizationResult(
                    image_name=frame_name,
                    sgil_err_m=sgil_err_m,
                    gps_err_m=float(
                        math.hypot(
                            curr_gps_xy[0] - rtk_pose.x,
                            curr_gps_xy[1] - rtk_pose.y,
                        )
                    ),
                    rtk_pose=rtk_pose_for_table,
                    current_pose=pose_for_match,
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

        print(
            f"\n{'frame':40s} | {'matched':10s} | {'sgil_err_m':10s} | {'gps_err_m':10s} | {'gps_pose':32s} | "
            f"{'rtk_pose':32s} | {'current_pose':32s} | {'estimated_pose':32s}"
        )
        print("-" * 40 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" "-" *32 + "-+-" + "-" *32 + "-+-" "-" * 32 + "-+-" + "-" * 32)

        for r in results:
            print(
                f"{r.image_name:40s} | "
                f"{r.matched!s:10s} | "
                f"{r.sgil_err_m:10.2f} | "
                f"{r.gps_err_m:10.2f} | "
                f"{fmt_pose(r.gps_pose) if r.gps_pose else 'None':32s} | "
                f"{fmt_pose(r.rtk_pose):32s} | "
                f"{fmt_pose(r.current_pose):32s} | "
                f"{fmt_pose(r.estimated_pose) if r.estimated_pose else 'None':32s}"
            )


if __name__ == "__main__":
    AutomatedSGIL().run()
