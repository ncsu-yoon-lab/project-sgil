"""Automated tree selection & localization from a labeled CSV.

file: automated_selector.py
author: Jack Elia
"""

from __future__ import annotations

import ast
import os

import pandas as pd
from constants import (
    CAMERA_HEIGHT_M,
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
)
from data_structs import LocalizationResult, Point, Pose2d

from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.imu_analyzer import IMUAnalyzer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter


class AutomatedSGIL:
    """Runs SGIL matching using a labeled CSV of tree pixel locations."""

    def __init__(
        self,
        labeled_csv_path: str = "../dataset/tables/first18images.csv",
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
    ) -> None:
        """Initialize state and load data.

        :param labeled_csv_path: Path to CSV with image filenames and
            tree pixel points.
        :param image_folder: Folder containing images (kept for parity /
            plotting).
        :param data_log_path: Robot data log CSV including
            image_filename, rtk_lat, rtk_lon, rtk_heading, gps_lat,
            gps_lon.
        :param origin: (lat, lon) origin for local frame conversion.
        """
        self.converter: Converter = Converter(origin[0], origin[1])
        self.tree_matcher: TreeMatcher = TreeMatcher()
        self.image_folder: str = image_folder
        self.image_shape = IMAGE_SHAPE
        self.camera_height_m = CAMERA_HEIGHT_M

        # Data
        self.robot_data_log: pd.DataFrame = pd.read_csv(data_log_path)
        self.labeled_data: pd.DataFrame = pd.read_csv(labeled_csv_path)

        # IMU analyzer (uses the same CSV as the robot log)
        self.imu_analyzer: IMUAnalyzer = IMUAnalyzer(data_log_path)

        # Runtime state
        self.current_pose: Pose2d | None = None
        self._last_row_index: int | None = None
        self._correct_pose_latlon: tuple[float, float] | None = None
        self._gps_pose_latlon: tuple[float, float] | None = None
        self._last_rtk_xy: tuple[float, float] | None = None  # handy for the diagnostic block

    def run(self) -> list[LocalizationResult]:
        """Process each labeled image and compute localization results.

        :return: List of LocalizationResult objects for the final
            summary table.
        """
        results: list[LocalizationResult] = []
        if PLOT:
            DebugVisualizer.clear_plots()

        for _, row in self.labeled_data.iterrows():
            image_name: str = str(row.get("image_filename", "")).strip()
            if not image_name:
                continue

            rtk_pose: Pose2d | None = self.get_current_pose(image_name)
            if rtk_pose is None:
                # Skip if no RTK pose available for this image
                continue

            # Map image -> row index in the robot log (one row per image as you noted)
            row_index = self._row_index_for_image(image_name)
            if row_index is None:
                continue

            # Initialize current_pose from RTK on the very first image
            # Subsequent images: update current_pose using IMU deltas
            if self.current_pose is None:
                self.current_pose = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=rtk_pose.yaw)
                self._last_row_index = row_index
            else:
                # Integrate IMU deltas from last row to current row (inclusive)
                period_len = row_index - int(self._last_row_index) + 1
                deltas = self.imu_analyzer.get_position_yaw_change(
                    start_index=int(self._last_row_index),
                    period_length=period_len,
                    sensor_type="main",
                    prefer_quaternion_yaw=True,
                )
                self.current_pose.x += rtk_pose.x - self._last_rtk_xy[0]
                self.current_pose.y += rtk_pose.y - self._last_rtk_xy[1]
                self.current_pose.yaw += float(deltas["delta_yaw_deg"])
                self._last_row_index = row_index
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            # Parse pixel points for this image; skip if none
            pixel_points = self.parse_tree_points(str(row.get("tree_points", "")).strip())
            if not pixel_points:
                continue

            # Compute ground-view thetas (deg) from pixel x
            ground_thetas = self.make_ground_thetas(pixel_points)

            # Use the pose passed in to the matcher (copy so table shows the exact input)
            pose_for_match = Pose2d(
                x=self.current_pose.x, y=self.current_pose.y, yaw=self.current_pose.yaw
            )
            # pose_for_match = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=self.current_pose.yaw)

            # Match and get estimated XY
            est_xy: Point = self.tree_matcher.match_trees(pose_for_match, ground_thetas)

            # Estimated pose keeps current yaw, replaces XY with estimated XY
            estimated_pose = Pose2d(x=est_xy.x, y=est_xy.y, yaw=self.current_pose.yaw)

            # Update current_pose's XY from the estimate
            self.current_pose.x = est_xy.x
            self.current_pose.y = est_xy.y

            # Compute SGIL error against RTK ground truth (meters)
            sgil_err_m = self._sgil_error_meters(estimated_pose)

            if PLOT:
                save_name: str = os.path.splitext(image_name)[0]
                DebugVisualizer.plot_aoi(
                    self.tree_matcher.satellite_tree_locations,
                    self.tree_matcher.aoi_trees,
                    rtk_pose,
                    save_name,
                )

            # Record table row
            results.append(
                LocalizationResult(
                    image_name=image_name,
                    sgil_err_m=sgil_err_m,
                    rtk_pose=rtk_pose,
                    current_pose=pose_for_match,
                    estimated_pose=estimated_pose,
                )
            )

        # Print a compact summary table
        self._print_summary_table(results)
        return results

    def get_current_pose(self, image_name: str) -> Pose2d | None:
        """Lookup the RTK/GPS pose row corresponding to the given image and
        convert to Pose2d.

        :param image_name: Image filename to search for.
        :return: Pose2d if rtk_heading is present; otherwise None.
        """
        df = self.robot_data_log
        matches = df[df["image_filename"].str.contains(image_name, case=False, na=False)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        if pd.isna(row.get("rtk_heading")):
            return None
        return self._row_to_pose(row)

    def make_ground_thetas(self, pixel_points: list[Point]) -> list[float]:
        """Convert image pixel x-coordinates to ground-view angles (degrees).

        :param pixel_points: Points in image pixel space.
        :return: List of ground angles (degrees), positive = left,
            negative = right.
        """
        thetas: list[float] = []
        for pt in pixel_points:
            theta = self.converter.image_x_to_theta(pt.x)
            thetas.append(theta)
        return thetas

    def _row_to_pose(self, row: pd.Series) -> Pose2d:
        """Convert a log row into a Pose2d and store lat/lon ground truth for
        error metrics.

        :param row: DataFrame row containing rtk_lat, rtk_lon,
            rtk_heading, gps_lat, gps_lon.
        :return: Pose2d in local XY plus yaw (degrees).
        """
        lat_rtk = float(row["rtk_lat"])
        lon_rtk = float(row["rtk_lon"])
        yaw_deg = self.converter.heading_to_yaw(float(row["rtk_heading"]))
        x, y = self.converter.latlon_to_xy((lat_rtk, lon_rtk))

        # Save for error metrics
        self._correct_pose_latlon = (lat_rtk, lon_rtk)
        self._gps_pose_latlon = (float(row["gps_lat"]), float(row["gps_lon"]))

        return Pose2d(x=x, y=y, yaw=yaw_deg)

    def _row_index_for_image(self, image_name: str) -> int | None:
        """Get the index of the robot log row that matches an image filename.

        :param image_name: Image filename to search.
        :return: Integer index if found; otherwise None.
        """
        matches = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        return int(matches.index[0])

    def _sgil_error_meters(self, estimated_pose: Pose2d) -> float:
        """Compute SGIL error in meters between an estimated pose and RTK
        ground truth.

        :param estimated_pose: Pose with XY in local frame.
        :return: Great-circle distance (meters) between estimated and
            RTK lat/lon.
        """
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
        """Parse the CSV 'tree_points' cell into a list of Points.

        :param value: String like "[(600, 343), (630, 341), ...]" or a sentinel like "NO PATH".
        :return: List of Point objects (image pixel coordinates).
        """
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
        """Print a compact summary table for all processed images.

        Columns: image_name, sgil_err_m, rtk_pose, current_pose, estimated_pose.
        """
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
