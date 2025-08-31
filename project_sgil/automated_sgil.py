"""
Automated tree selection & localization from a labeled CSV.

file: automated_selector.py
author: Jack Elia
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from constants import (
    IMAGE_FOLDER_PATH,
    DATA_LOGGER_PATH,
    ORIGIN,
    IMAGE_SHAPE,
    CAMERA_HEIGHT_M,
    PLOT,
)
from data_structs import Point, Pose2d
from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter


logging.basicConfig(level=logging.INFO)


@dataclass
class LocalizationResult:
    """Container for per-image localization results.

    :param image_name: Image filename processed.
    :param est_xy: Estimated XY position in local map frame.
    :param est_latlon: Estimated latitude/longitude.
    :param gps_error_m: Distance (m) between GPS and RTK ground truth.
    :param sgil_error_m: Distance (m) between SGIL estimate and RTK ground truth.
    :param rtk_yaw_deg: RTK yaw (deg) used for the pose.
    """
    image_name: str
    est_xy: Point
    est_latlon: Tuple[float, float]
    gps_error_m: float
    sgil_error_m: float
    rtk_yaw_deg: float


class AutomatedSGIL:
    """Runs SGIL matching using a labeled CSV of tree pixel locations.

    Workflow:
      1) Loads robot RTK/GPS log (for pose & ground truth).
      2) Loads a labeled CSV with image filenames and tree pixel coordinates.
      3) Converts pixel x to ground angles (theta) for each image.
      4) Calls TreeMatcher to estimate XY.
      5) Reports GPS vs RTK error and SGIL vs RTK error.

    Expected CSV columns:
      - image_filename : str (e.g., "image_1746551059_931153577.jpg")
      - tree_points    : str representation of a Python list of (x, y) tuples.
                         Example: "[(600, 343), (630, 341), ...]".
    """

    def __init__(
        self,
        labeled_csv_path: str = "../dataset/tables/first18images.csv",
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: Tuple[float, float] = ORIGIN,
    ) -> None:
        """Initialize state and load data.

        :param labeled_csv_path: Path to CSV with image filenames and tree pixel points.
        :param image_folder: Folder containing images (not strictly required here, kept for parity).
        :param data_log_path: Robot data log CSV including image_filename, rtk_lat, rtk_lon, rtk_heading, gps_lat, gps_lon.
        :param origin: (lat, lon) origin for local frame conversion.
        """
        self.converter: Converter = Converter(origin[0], origin[1])
        self.tree_matcher: TreeMatcher = TreeMatcher()
        self.image_folder: str = image_folder
        self.image_shape = IMAGE_SHAPE
        self.camera_height_m = CAMERA_HEIGHT_M

        # Load datasets once
        self.robot_data_log: pd.DataFrame = pd.read_csv(data_log_path)
        self.labeled_data: pd.DataFrame = pd.read_csv(labeled_csv_path)

        # Internal, updated per row for error reporting
        self._correct_pose_latlon: Optional[Tuple[float, float]] = None
        self._gps_pose_latlon: Optional[Tuple[float, float]] = None

    # --------------------------
    # Public API
    # --------------------------

    def run(self) -> List[LocalizationResult]:
        """Process each labeled image and compute localization results."""
        print("Starting automated SGIL from labeled CSV...")
        print("- Using pre-labeled tree pixel coordinates")
        print("- Converting pixel x → ground theta")
        print("- Matching with satellite trees via TreeMatcher\n")

        results: List[LocalizationResult] = []
        DebugVisualizer.clear_plots()

        for _, row in self.labeled_data.iterrows():
            image_name: str = str(row.get("image_filename", "")).strip()
            if not image_name:
                logging.warning("Skipping row with empty image_filename.")
                continue

            pose: Optional[Pose2d] = self.get_current_pose(image_name)
            if pose is None:
                logging.warning(f"No valid RTK pose for {image_name}; skipping.")
                continue

            tree_points_str: str = str(row.get("tree_points", "")).strip()
            pixel_points: List[Point] = self.parse_tree_points(tree_points_str)
            if len(pixel_points) == 0:
                logging.warning(f"No usable tree points for {image_name}; skipping.")
                continue

            tree_matcher = TreeMatcher()

            ground_thetas: List[float] = self.make_ground_thetas(pixel_points)
            est_xy: Point = tree_matcher.match_trees(pose, ground_thetas)
            est_latlon: Tuple[float, float] = self.converter.xy_to_latlon(est_xy)

            if self._correct_pose_latlon is None or self._gps_pose_latlon is None:
                logging.error("Pose info missing for error computation; skipping metrics.")
                continue

            gps_err: float = self.converter.haversine(
                self._gps_pose_latlon[0],
                self._gps_pose_latlon[1],
                self._correct_pose_latlon[0],
                self._correct_pose_latlon[1],
            )
            sgil_err: float = self.converter.haversine(
                est_latlon[0],
                est_latlon[1],
                self._correct_pose_latlon[0],
                self._correct_pose_latlon[1],
            )

            logging.info(f"Image: {image_name}")
            logging.info(f"  Estimated LatLon: {est_latlon}")
            logging.info(f"  GPS error (m):     {gps_err:.2f}")
            logging.info(f"  SGIL error (m):    {sgil_err:.2f}")
            logging.info(f"  RTK Yaw (deg):     {pose.yaw:.2f}")

            if PLOT:
                save_name: str = os.path.splitext(image_name)[0]
                DebugVisualizer.plot_aoi(
                    tree_matcher.satellite_tree_locations,
                    tree_matcher.aoi_trees,
                    pose,
                    save_name,
                )

            results.append(
                LocalizationResult(
                    image_name=image_name,
                    est_xy=est_xy,
                    est_latlon=est_latlon,
                    gps_error_m=gps_err,
                    sgil_error_m=sgil_err,
                    rtk_yaw_deg=pose.yaw,
                )
            )

        return results

    # --------------------------
    # Helpers (no nesting)
    # --------------------------

    def parse_tree_points(self, value: str) -> List[Point]:
        """Parse the CSV 'tree_points' cell into a list of Points.

        :param value: String like "[(600, 343), (630, 341), ...]" or a sentinel like "NO PATH".
        :return: List of Point objects (image pixel coordinates).
        """
        points: List[Point] = []

        if not value:
            return points

        upper = value.upper()
        if "NO PATH" in upper or "NONE" in upper or upper == "NAN":
            return points

        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            logging.warning(f"Could not parse tree_points: {value!r}")
            return points

        if not isinstance(parsed, (list, tuple)):
            return points

        for item in parsed:
            # Expect (x, y) pairs
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    x = float(item[0])
                    y = float(item[1])
                    points.append(Point(x=x, y=y))
                except (TypeError, ValueError):
                    continue

        return points

    def make_ground_thetas(self, pixel_points: List[Point]) -> List[float]:
        """Convert image pixel x-coordinates to ground-view angles.

        :param pixel_points: Points in image pixel space.
        :return: List of ground angles (degrees), positive = left, negative = right.
        """
        thetas: List[float] = []
        for pt in pixel_points:
            theta = self.converter.image_x_to_theta(pt.x)
            thetas.append(theta)
        return thetas

    def get_current_pose(self, image_name: str) -> Optional[Pose2d]:
        """Lookup the RTK/GPS pose row corresponding to the given image and convert to Pose2d.

        :param image_name: Image filename to search for.
        :return: Pose2d if rtk_heading is present; otherwise None.
        """
        df = self.robot_data_log
        # Use contains for robustness in case paths are included
        matches = df[df["image_filename"].str.contains(image_name, case=False, na=False)]
        if matches.empty:
            return None

        row = matches.iloc[0]
        if pd.isna(row.get("rtk_heading")):
            return None

        return self._row_to_pose(row)

    def _row_to_pose(self, row: pd.Series) -> Pose2d:
        """Convert a log row into a Pose2d, storing lat/lon fields for later error metrics.

        :param row: DataFrame row containing rtk_lat, rtk_lon, rtk_heading, gps_lat, gps_lon.
        :return: Pose2d in local XY plus yaw (deg).
        """
        lat_rtk = float(row["rtk_lat"])
        lon_rtk = float(row["rtk_lon"])
        yaw_deg = self.converter.heading_to_yaw(float(row["rtk_heading"]))

        x, y = self.converter.latlon_to_xy((lat_rtk, lon_rtk))

        # Save for error metrics
        self._correct_pose_latlon = (lat_rtk, lon_rtk)
        self._gps_pose_latlon = (float(row["gps_lat"]), float(row["gps_lon"]))

        return Pose2d(x=x, y=y, yaw=yaw_deg)


if __name__ == "__main__":
    results = AutomatedSGIL().run()
    # Print a compact summary table
    if len(results) > 0:
        print("\nSummary:")
        for r in results:
            print(
                f"{r.image_name:40s} | GPS err: {r.gps_error_m:7.2f} m | "
                f"SGIL err: {r.sgil_error_m:7.2f} m | RTK yaw: {r.rtk_yaw_deg:7.2f}°"
            )
    else:
        print("No images processed.")
