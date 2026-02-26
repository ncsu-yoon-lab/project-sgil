"""Automated tree selection & localization from a labeled CSV.

file: automated_selector.py
author: Jack Elia
"""

from __future__ import annotations

import ast
import math
import os
import statistics

import pandas as pd
from constants import (
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
)
from data_structs import LocalizationResult, Point, Pose2d

from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter


class AutomatedSGIL:
    """Runs SGIL matching using a labeled CSV of tree pixel locations."""

    def __init__(
        self,
        labeled_csv_path: str = "../dataset/tables/annotations_interactive.csv",
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
    ) -> None:
        self.converter: Converter = Converter(origin[0], origin[1])
        self.tree_matcher: TreeMatcher = TreeMatcher()
        self.image_folder: str = image_folder
        self.image_shape = IMAGE_SHAPE

        # Data
        self.robot_data_log: pd.DataFrame = pd.read_csv(data_log_path)
        self.labeled_data: pd.DataFrame = pd.read_csv(labeled_csv_path)

        # Runtime state
        self.current_pose: Pose2d | None = None
        self._last_row_index: int | None = None
        self._correct_pose_latlon: tuple[float, float] | None = None
        self._last_rtk_xy: tuple[float, float] | None = None  # for RTK deltas

    def run(self) -> list[LocalizationResult]:
        """Process each labeled image and compute localization results."""
        results: list[LocalizationResult] = []
        if PLOT:
            DebugVisualizer.clear_plots()

        for _, row in self.labeled_data.iterrows():
            self.tree_matcher = TreeMatcher(False)

            image_name: str = str(row.get("image_filename", "")).strip()
            if not image_name:
                continue

            if image_name == "image_1746551245_831563501.jpg":
                print(f"\nStopping early at {image_name}")
                break

            rtk_pose: Pose2d | None = self.get_current_pose(image_name)
            if rtk_pose is None:
                continue

            row_index = self._row_index_for_image(image_name)
            if row_index is None:
                continue

            if self.current_pose is None:
                self.current_pose = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=rtk_pose.yaw)
                self._last_row_index = row_index
                self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)
            else:
                self.current_pose.yaw = rtk_pose.yaw
                self._last_row_index = row_index

            # RTK Δx,Δy since last anchor (handles skipped images)
            if self._last_rtk_xy is None:
                dx_rtk = dy_rtk = 0.0
            else:
                dx_rtk = rtk_pose.x - self._last_rtk_xy[0]
                dy_rtk = rtk_pose.y - self._last_rtk_xy[1]

            # Predicted XY = last estimated XY + RTK ΔXY
            predicted_x = self.current_pose.x + dx_rtk
            predicted_y = self.current_pose.y + dy_rtk

            # Parse tree points
            pixel_points = self.parse_tree_points(str(row.get("tree_points", "")).strip())

            if pixel_points:
                ground_thetas = self.make_ground_thetas(pixel_points)
                # Pose used for matching: predicted XY + RTK yaw
                pose_for_match = Pose2d(x=predicted_x, y=predicted_y, yaw=self.current_pose.yaw)

                # Try TreeMatcher; on failure, fall back to predicted
                try:
                    # pose_for_match = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=self.current_pose.yaw)
                    est_xy: Point = self.tree_matcher.match_trees(pose_for_match, ground_thetas)
                    estimated_pose = Pose2d(x=est_xy.x, y=est_xy.y, yaw=self.current_pose.yaw)
                except Exception:
                    # Fallback: no change beyond RTK ΔXY
                    estimated_pose = Pose2d(x=predicted_x, y=predicted_y, yaw=self.current_pose.yaw)
                    pose_for_match = Pose2d(x=predicted_x, y=predicted_y, yaw=self.current_pose.yaw)
            else:
                # No trees: just use predicted pose
                pose_for_match = Pose2d(x=predicted_x, y=predicted_y, yaw=self.current_pose.yaw)
                estimated_pose = Pose2d(x=predicted_x, y=predicted_y, yaw=self.current_pose.yaw)

            # Advance state with the chosen estimate and update RTK anchor
            self.current_pose = Pose2d(
                x=estimated_pose.x, y=estimated_pose.y, yaw=self.current_pose.yaw
            )
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            # Error vs RTK
            sgil_err_m = self._sgil_error_meters(estimated_pose)

            if PLOT:
                save_name: str = os.path.splitext(image_name)[0]
                DebugVisualizer.plot_aoi(
                    self.tree_matcher.satellite_tree_locations,
                    self.tree_matcher.aoi_trees,
                    rtk_pose,
                    save_name,
                )

            results.append(
                LocalizationResult(
                    image_name=image_name,
                    sgil_err_m=sgil_err_m,
                    rtk_pose=rtk_pose,
                    current_pose=pose_for_match,
                    estimated_pose=estimated_pose,
                )
            )

        if results:
            valid_errors = [
                r.sgil_err_m
                for r in results
                if pd.notna(r.sgil_err_m) and math.isfinite(r.sgil_err_m)
            ]
            if valid_errors:
                mean_err = sum(valid_errors) / len(valid_errors)
                median_err = statistics.median(valid_errors)
                print(f"\nMean SGIL error   over {len(valid_errors)} images: {mean_err:.3f} m")
                print(f"Median SGIL error over {len(valid_errors)} images: {median_err:.3f} m")
            else:
                print("\nNo valid SGIL errors to compute statistics.")

        self._print_summary_table(results)
        return results

    def get_current_pose(self, image_name: str) -> Pose2d | None:
        """Lookup the RTK/GPS pose row corresponding to the given image and
        convert to Pose2d."""
        df = self.robot_data_log
        matches = df[df["image_filename"].str.contains(image_name, case=False, na=False)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        if pd.isna(row.get("rtk_heading")):
            return None
        return self._row_to_pose(row)

    def make_ground_thetas(self, pixel_points: list[Point]) -> list[float]:
        thetas: list[float] = []
        for pt in pixel_points:
            thetas.append(self.converter.image_x_to_theta(pt.x))
        return thetas

    def _row_to_pose(self, row: pd.Series) -> Pose2d:
        """Note: using GPS lat/lon here, per your current code."""
        lat_rtk = float(row["rtk_lat"])
        lon_rtk = float(row["rtk_lon"])
        yaw_deg = self.converter.heading_to_yaw(float(row["rtk_heading"]))
        x, y = self.converter.latlon_to_xy((lat_rtk, lon_rtk))

        # Save for error metrics
        self._correct_pose_latlon = (lat_rtk, lon_rtk)

        return Pose2d(x=x, y=y, yaw=yaw_deg)

    def _row_index_for_image(self, image_name: str) -> int | None:
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
    def parse_tree_points(value: str) -> list[Point]:
        points: list[Point] = []
        if not value:
            return points

        upper = value.upper()
        if "NO PATH" in upper or "NONE" in upper or upper == "NAN":
            return points

        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return points

        if not isinstance(parsed, list | tuple):
            return points

        for item in parsed:
            if isinstance(item, list | tuple) and len(item) == 2:
                try:
                    x = float(item[0])
                    y = float(item[1])
                    points.append(Point(x=x, y=y))
                except (TypeError, ValueError):
                    continue

        return points

    def _print_summary_table(self, results: list[LocalizationResult]) -> None:
        if not results:
            print("No images processed.")
            return

        def fmt_pose(p: Pose2d) -> str:
            return f"(x={p.x:.2f}, y={p.y:.2f}, yaw={p.yaw:.2f}°)"

        col1, col2, col3, col4, col5 = (
            "image_name",
            "sgil_err_m",
            "rtk_pose",
            "current_pose",
            "estimated_pose",
        )
        print(f"\n{col1:40s} | {col2:10s} | {col3:32s} | {col4:32s} | {col5:32s}")
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
    results = AutomatedSGIL().run()
