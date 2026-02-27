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
    AUTOMATED_FRAME_STEP,
    AUTOMATED_MIN_SEGMENT_CONFIDENCE,
    AUTOMATED_SKIP_IF_NO_TREES,
    AUTOMATED_USE_RTK_POSE_EACH_FRAME,
    DATA_LOGGER_PATH,
    HEADING_SWEEP_ENABLED,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
    PLOT_WEDGES,
    RTK_TO_CAMERA_OFFSET_X_FWD_M,
    RTK_TO_CAMERA_OFFSET_Y_LEFT_M,
)
from project_sgil.data_structs import LocalizationResult, Point, Pose2d
from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter


class AutomatedSGIL:
    """Runs SGIL matching using segmentation centroids from results.json."""

    def __init__(
        self,
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
    ) -> None:
        self.converter = Converter(origin[0], origin[1])
        self.tree_matcher = TreeMatcher(PLOT_WEDGES)
        self.image_folder = image_folder
        self.image_shape = IMAGE_SHAPE

        # Load JSON frames once
        self.data_log_path = self._resolve_json_path(data_log_path)
        self.frame_lookup = self._load_frame_lookup(self.data_log_path)

        # Runtime state
        self.current_pose: Pose2d | None = None
        self._last_rtk_xy: tuple[float, float] | None = None
        self._correct_pose_latlon: tuple[float, float] | None = None
        self._rtk_pose_latlon: tuple[float, float] | None = None
        self._gps_pose_latlon: tuple[float, float] | None = None

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

    @staticmethod
    def _frame_selected(idx: int) -> bool:
        if idx < AUTOMATED_FRAME_START or idx > AUTOMATED_FRAME_END:
            return False
        if AUTOMATED_FRAME_STEP <= 1:
            return True
        return (idx - AUTOMATED_FRAME_START) % AUTOMATED_FRAME_STEP == 0

    # ---------------- Pose + trees ----------------

    def _frame_to_pose(self, frame: dict[str, Any]) -> Pose2d | None:
        if frame.get("rtk_heading") is None:
            return None

        # Store lat/lon (raw antenna locations)
        self._rtk_pose_latlon = (float(frame["rtk_lat"]), float(frame["rtk_lon"]))
        self._gps_pose_latlon = (float(frame["gps_lat"]), float(frame["gps_lon"]))

        # RTK antenna position in XY
        x, y = self.converter.latlon_to_xy(self._rtk_pose_latlon)

        # RTK yaw
        yaw = self.converter.heading_to_yaw(frame["rtk_heading"]) + 90.0 - 15.0

        # Apply extrinsic: RTK -> camera (body frame offsets rotated by yaw)
        dx_b = float(RTK_TO_CAMERA_OFFSET_X_FWD_M)
        dy_b = float(RTK_TO_CAMERA_OFFSET_Y_LEFT_M)
        c = math.cos(math.radians(yaw))
        s = math.sin(math.radians(yaw))
        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        x = x - dx_w
        y = y - dy_w

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
        return float(math.hypot(estimated_pose.x - rtk_pose.x, estimated_pose.y - rtk_pose.y))

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

            rtk_pose = self._frame_to_pose(frame)
            if rtk_pose is None:
                continue

            pixel_points = self._pixel_points_from_segmentations(frame)
            if AUTOMATED_SKIP_IF_NO_TREES and not pixel_points:
                continue

            # Initialize state
            if self.current_pose is None:
                self.current_pose = Pose2d(rtk_pose.x, rtk_pose.y, rtk_pose.yaw)
                self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            assert self.current_pose is not None

            if AUTOMATED_USE_RTK_POSE_EACH_FRAME:
                # No carryover: use RTK pose directly for matching.
                predicted_x = rtk_pose.x
                predicted_y = rtk_pose.y
            else:
                # RTK delta since last processed
                if self._last_rtk_xy is None:
                    dx_rtk = dy_rtk = 0.0
                else:
                    dx_rtk = rtk_pose.x - self._last_rtk_xy[0]
                    dy_rtk = rtk_pose.y - self._last_rtk_xy[1]

                predicted_x = self.current_pose.x + dx_rtk
                predicted_y = self.current_pose.y + dy_rtk

            # Convert pixel points -> ground thetas
            img_w = int(self.image_shape[1])
            ground_thetas = [
                self.converter.image_x_to_theta(pt.x, img_w) for pt in pixel_points
            ]

            pose_for_match = Pose2d(predicted_x, predicted_y, rtk_pose.yaw)

            try:
                est_pose = self.tree_matcher.match_trees(pose_for_match, ground_thetas)
                if not isinstance(est_pose, Pose2d):
                    est_pose = Pose2d(est_pose.x, est_pose.y, pose_for_match.yaw)
            except Exception:
                est_pose = Pose2d(predicted_x, predicted_y, pose_for_match.yaw)

            # If heading sweep is disabled, keep using RTK yaw
            if not HEADING_SWEEP_ENABLED:
                est_pose.yaw = pose_for_match.yaw

            # Advance
            if AUTOMATED_USE_RTK_POSE_EACH_FRAME:
                # Keep the internal state in sync with RTK if we're in pure-RTK mode.
                self.current_pose = Pose2d(rtk_pose.x, rtk_pose.y, rtk_pose.yaw)
            else:
                self.current_pose = Pose2d(est_pose.x, est_pose.y, est_pose.yaw)
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            sgil_err_m = self._sgil_error_meters(est_pose, rtk_pose)

            if PLOT:
                save_stub = f"frame_{idx:06d}"
                DebugVisualizer.plot_aoi(
                    self.tree_matcher.satellite_tree_locations,
                    self.tree_matcher.aoi_trees,
                    rtk_pose,
                    save_stub,
                )

            results.append(
                LocalizationResult(
                    image_name=frame_name,
                    sgil_err_m=sgil_err_m,
                    rtk_pose=rtk_pose,
                    current_pose=pose_for_match,
                    estimated_pose=est_pose,
                )
            )

        if results:
            valid_errors = [r.sgil_err_m for r in results if pd.notna(r.sgil_err_m)]
            if valid_errors:
                mean_err = sum(valid_errors) / len(valid_errors)
                median_err = statistics.median(valid_errors)
                print(f"\nMean SGIL error   over {len(valid_errors)} frames: {mean_err:.3f} m")
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
            f"\n{'frame':40s} | {'sgil_err_m':10s} | {'rtk_pose':32s} | "
            f"{'current_pose':32s} | {'estimated_pose':32s}"
        )
        print("-" * 40 + "-+-" + "-" * 10 + "-+-" + "-" * 32 + "-+-" + "-" * 32 + "-+-" + "-" * 32)

        for r in results:
            print(
                f"{r.image_name:40s} | "
                f"{r.sgil_err_m:10.2f} | "
                f"{fmt_pose(r.rtk_pose):32s} | "
                f"{fmt_pose(r.current_pose):32s} | "
                f"{fmt_pose(r.estimated_pose):32s}"
            )


if __name__ == "__main__":
    AutomatedSGIL().run()
