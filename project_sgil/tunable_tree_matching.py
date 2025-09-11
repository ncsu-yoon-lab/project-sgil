# automated_sgil_tuner.py

from __future__ import annotations

import ast
import itertools
import math
import os
from typing import Optional

import pandas as pd

from constants import (
    CAMERA_HEIGHT_M,
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
    TREE_LOCATIONS_PATH,
)
from data_structs import Point, Pose2d
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.localization.imu_analyzer import IMUAnalyzer
from project_sgil.utils.converter import Converter


# ---------- Tunable matcher (only scoring changes) ----------

class TunableTreeMatcher(TreeMatcher):
    def __init__(
        self,
        plot_wedges: bool = False,
        path: str = TREE_LOCATIONS_PATH,
        w_occ: float = 0.6,
        w_rms: float = 0.3,
        w_theta: float = 0.3,
        w_num: float = 0.5,
    ) -> None:
        super().__init__(plot_wedges=plot_wedges, path=path)
        self.w_occ = float(w_occ)
        self.w_rms = float(w_rms)
        self.w_theta = float(w_theta)
        self.w_num = float(w_num)

    # Override to use instance weights instead of constants.
    def _calculate_pose_estimate_score(
        self,
        wedge_map,  # dict[Wedge, Tree]
        estimated_pose: Pose2d,
        residual_rms: float,
    ) -> float:
        if not wedge_map:
            return 0.0

        visible_count = 0
        theta_match_count = 0
        total_selected = len(wedge_map)

        # Occlusion & theta checks
        for wedge, selected_tree in wedge_map.items():
            # (a) occlusion
            occluded = False
            for other_tree in self.aoi_trees:
                if other_tree.id == selected_tree.id:
                    continue
                # Reuse existing util via DebugVisualizer module imports in original class
                from project_sgil.utils.utils import _segment_intersects_circle
                from project_sgil.constants import TREE_RADIUS_M

                if _segment_intersects_circle(
                    start=estimated_pose,
                    end=selected_tree,
                    center=other_tree,
                    radius=TREE_RADIUS_M,
                ):
                    occluded = True
                    break
            if not occluded:
                visible_count += 1

            # (b) theta matching
            from project_sgil.utils.utils import get_relative_angle
            from project_sgil.constants import THETA_MATCHING_TOLERANCE

            observed_deg = get_relative_angle(selected_tree, estimated_pose)
            diff = (observed_deg - wedge.theta_degrees + 180.0) % 360.0 - 180.0
            if abs(diff) <= THETA_MATCHING_TOLERANCE:
                theta_match_count += 1

        visibility_ratio = visible_count / float(total_selected)
        theta_match_ratio = theta_match_count / float(total_selected)

        # RMS term (only meaningful if 3+ trees)
        if total_selected <= 2:
            rms_component = 0.0
        else:
            # Same “soft” RMS normalization as original
            from project_sgil.constants import SCORE_RMS_SCALE
            rms_score = 1.0 / (1.0 + (residual_rms / max(1e-9, SCORE_RMS_SCALE)))
            rms_component = self.w_rms * rms_score

        # Combine
        score = (
            self.w_occ * visibility_ratio
            + self.w_theta * theta_match_ratio
            + rms_component
            + self.w_num * total_selected
        )
        return float(score)


# ---------- Tuning runner ----------

class AutomatedSGILTuner:
    """Grid-search the four score weights and report the best by mean SGIL error."""

    def __init__(
        self,
        labeled_csv_path: str = "../dataset/tables/annotations_interactive.csv",
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
    ) -> None:
        self.converter: Converter = Converter(origin[0], origin[1])
        self.image_folder: str = image_folder
        self.image_shape = IMAGE_SHAPE
        self.camera_height_m = CAMERA_HEIGHT_M

        # Data
        self.robot_data_log: pd.DataFrame = pd.read_csv(data_log_path)
        self.labeled_data: pd.DataFrame = pd.read_csv(labeled_csv_path)

        # IMU (yaw-only usage)
        self.imu_analyzer: IMUAnalyzer = IMUAnalyzer(self.robot_data_log)

        # Runtime state across an evaluation pass
        self.current_pose: Optional[Pose2d] = None
        self._last_row_index: Optional[int] = None
        self._last_rtk_xy: Optional[tuple[float, float]] = None
        self._correct_pose_latlon: Optional[tuple[float, float]] = None

    # --------- Public entry ---------

    def run(self) -> None:
        values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        best = None  # (mean_err, (w_occ, w_rms, w_theta, w_num))

        # Iterate all combos (fix w_num=0.5 for now)
        combos = itertools.product(values, values, values)
        i = 0
        for (w_occ, w_rms, w_theta) in combos:
            i += 1
            print(f"Working on combo {i}. Tuning percent complete: {100.0 * i / (len(values)**3):.1f}%")
            mean_err = self._evaluate_combo(w_occ, w_rms, w_theta, 0.6)
            if mean_err is None:
                # No successful matches at all for this combo — discard
                continue
            if best is None or mean_err < best[0]:
                best = (mean_err, (w_occ, w_rms, w_theta, 0.6))

        if best is None:
            print("No weight combination produced any successful matches.")
            return

        mean_err, (w_occ, w_rms, w_theta, w_num) = best
        print("\nBest weights (by mean SGIL error):")
        print(f"  SCORE_WEIGHT_OCCLUSION = {w_occ:.1f}")
        print(f"  SCORE_WEIGHT_RMS       = {w_rms:.1f}")
        print(f"  SCORE_WEIGHT_THETA     = {w_theta:.1f}")
        print(f"  NUMBER_SELECTED_WEIGHT = {w_num:.1f}")
        print(f"Mean SGIL error (m): {mean_err:.3f}")

    # --------- One evaluation pass over all labeled images ---------

    def _evaluate_combo(self, w_occ: float, w_rms: float, w_theta: float, w_num: float) -> Optional[float]:
        """Run through labeled images once using a TunableTreeMatcher with given weights."""
        matcher = TunableTreeMatcher(
            plot_wedges=False,
            path=TREE_LOCATIONS_PATH,
            w_occ=w_occ,
            w_rms=w_rms,
            w_theta=w_theta,
            w_num=w_num,
        )

        # Reset state for this pass
        self.current_pose = None
        self._last_row_index = None
        self._last_rtk_xy = None

        errors: list[float] = []

        # Iterate labeled rows in order
        for _, row in self.labeled_data.iterrows():
            image_name: str = str(row.get("image_filename", "")).strip()
            if not image_name:
                continue

            if image_name == "image_1746551245_831563501.jpg":
                print(f"\nStopping early at {image_name}")
                break

            rtk_pose = self._get_rtk_pose(image_name)
            if rtk_pose is None:
                continue

            # Row index for IMU time window
            row_index = self._row_index_for_image(image_name)
            if row_index is None:
                continue

            # Maintain current yaw via IMU delta; keep XY deltas from RTK (as in your current flow)
            if self.current_pose is None:
                self.current_pose = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=rtk_pose.yaw)
                self._last_row_index = row_index
            else:
                # XY drift by RTK delta
                if self._last_rtk_xy is not None:
                    self.current_pose.x += rtk_pose.x - self._last_rtk_xy[0]
                    self.current_pose.y += rtk_pose.y - self._last_rtk_xy[1]
                # Yaw update by IMU delta (degrees)
                assert self._last_row_index is not None
                dyaw = self.imu_analyzer.delta_yaw_deg(int(self._last_row_index), row_index)
                self.current_pose.yaw += float(dyaw)
                self._last_row_index = row_index
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            # Parse tree clicks; skip if none
            pixel_points = self._parse_tree_points(str(row.get("tree_points", "")).strip())
            if not pixel_points:
                continue

            # thetas from pixel x
            ground_thetas = [self.converter.image_x_to_theta(pt.x) for pt in pixel_points]

            # Pose passed to matcher: RTK XY + IMU-updated yaw
            pose_for_match = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=self.current_pose.yaw)

            # Try to match; skip image on failure
            try:
                est_xy: Point = matcher.match_trees(pose_for_match, ground_thetas)
            except Exception:
                continue

            # Update state and compute error
            self.current_pose.x = est_xy.x
            self.current_pose.y = est_xy.y

            err_m = self._sgil_error_meters(Pose2d(est_xy.x, est_xy.y, self.current_pose.yaw))
            if math.isfinite(err_m):
                errors.append(err_m)

        if len(errors) == 0:
            return None
        return float(sum(errors) / len(errors))

    # --------- Helpers (same behaviors as your AutomatedSGIL) ---------

    def _get_rtk_pose(self, image_name: str) -> Optional[Pose2d]:
        matches = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        row = matches.iloc[0]
        if pd.isna(row.get("rtk_heading")):
            return None

        lat_rtk = float(row["rtk_lat"])
        lon_rtk = float(row["rtk_lon"])
        yaw_deg = self.converter.heading_to_yaw(float(row["rtk_heading"]))
        x, y = self.converter.latlon_to_xy((lat_rtk, lon_rtk))

        self._correct_pose_latlon = (lat_rtk, lon_rtk)
        return Pose2d(x=x, y=y, yaw=yaw_deg)

    def _row_index_for_image(self, image_name: str) -> Optional[int]:
        matches = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        return int(matches.index[0])

    def _sgil_error_meters(self, estimated_pose: Pose2d) -> float:
        if self._correct_pose_latlon is None:
            return float("nan")
        est_latlon = self.converter.xy_to_latlon(Point(estimated_pose.x, estimated_pose.y))
        return float(
            self.converter.haversine(
                est_latlon[0],
                est_latlon[1],
                self._correct_pose_latlon[0],
                self._correct_pose_latlon[1],
            )
        )

    @staticmethod
    def _parse_tree_points(value: str) -> list[Point]:
        points: list[Point] = []
        if not value:
            return points
        up = value.upper()
        if "NO PATH" in up or "NONE" in up or up == "NAN":
            return points
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return points
        if not isinstance(parsed, (list, tuple)):
            return points
        for it in parsed:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                try:
                    points.append(Point(float(it[0]), float(it[1])))
                except (TypeError, ValueError):
                    pass
        return points


if __name__ == "__main__":
    # You can tweak the default paths here if needed
    AutomatedSGILTuner(
        labeled_csv_path="../dataset/tables/annotations_interactive.csv",
        image_folder=IMAGE_FOLDER_PATH,
        data_log_path=DATA_LOGGER_PATH,
        origin=ORIGIN,
    ).run()
