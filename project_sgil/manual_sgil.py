"""Manual tree selection interface for ground view analysis and matching with
satellite data using pre- collected and pre-labeled dataset.

file: manual_selector.py
author: Cole Malinchock and Jack Elia
"""

# Import standard libraries
import json
import logging
import os
import random
from typing import Any
import math

import cv2

# Import custom classes
from project_sgil.constants import (
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
    PLOT_WEDGES,
    RANDOM,
    RTK_TO_CAMERA_OFFSET_X_FWD_M,
    RTK_TO_CAMERA_OFFSET_Y_LEFT_M,
)
from project_sgil.data_structs import Point, Pose2d

from matplotlib import pyplot as plt

from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter

logging.basicConfig(level=logging.INFO)


class ManualSGIL:
    """
    Encapsulates the SGIL tree-matching workflow:
      1. Reads robot GPS/RTK logs.
      2. Presents each image for the user to click tree locations.
      3. Converts clicks into ground angles.
      4. Matches trees via TreeMatcher.
      5. Computes and prints error metrics.
    """

    def __init__(
        self,
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
        randomize: bool = RANDOM,
    ) -> None:
        """
        :param image_folder: Path containing the .jpg images.
        :param data_log_path: JSON with frames containing
                              frame_name, rtk_lat, rtk_lon,
                              rtk_heading, gps_lat, gps_lon.
        :param origin: (lat, lon) of the converter origin.
        :param randomize: Whether to pick the next image at random.
        """
        self.converter = Converter(origin[0], origin[1])
        self.tree_matcher = TreeMatcher(PLOT_WEDGES)
        self.image_folder = image_folder
        self.randomize = randomize
        self.image_shape = IMAGE_SHAPE

        # Load robot log into a frame lookup once
        self.data_log_path = self._resolve_json_path(data_log_path)
        self.frame_lookup = self._load_frame_lookup(self.data_log_path)

        # Will be set on each call to get_gps_pose
        self.correct_pose_latlon: tuple[float, float] | None = None
        self.gps_pose_latlon: tuple[float, float] | None = None

        # Cached RTK yaw + offset rotation for error calculations
        self._last_rtk_yaw_deg: float | None = None

        # Prepare image list state
        self._image_exts = (".jpg", ".jpeg", ".png")
        self._image_list: list[str] = sorted(
            f
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(self._image_exts)
        )
        self._current_index: int = 0

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
            if name:
                lookup[name] = frame
                # Also store by basename to handle paths like "images/foo.png"
                lookup[os.path.basename(name)] = frame

        return lookup

    def get_current_pose(self, image_name: str) -> Pose2d | None:
        """Lookup the RTK/GPS pose for a given image filename.

        :param image_name: Name of the image.
        :return: Pose2d if RTK heading is valid; otherwise None.
        """
        frame = self._get_frame_for_image(image_name)
        if frame is None or frame.get("rtk_heading") is None:
            return None

        return self.get_gps_pose(frame)

    def _get_frame_for_image(self, image_name: str) -> dict[str, Any] | None:
        frame = self.frame_lookup.get(image_name)
        if frame is not None:
            return frame

        # Try basename match if image_name includes a path
        base_name = os.path.basename(image_name)
        frame = self.frame_lookup.get(base_name)
        if frame is not None:
            return frame

        # Try matching by basename if extensions differ (.jpg vs .png)
        base, _ = os.path.splitext(base_name)
        for ext in self._image_exts:
            candidate = f"{base}{ext}"
            frame = self.frame_lookup.get(candidate)
            if frame is not None:
                return frame

        return None

    def get_gps_pose(self, frame: dict[str, Any]) -> Pose2d:
        """Convert a log frame into a Pose2d using RTK for yaw.

        Also updates internal gps_pose/correct_pose for error logging.
        :param frame: dict with rtk_lat, rtk_lon, rtk_heading,
            gps_lat, gps_lon.
        :return: Pose2d in local XY + yaw degrees.
        """
        # Convert lat/lon to XY (RTK antenna position)
        x, y = self.converter.latlon_to_xy((frame["rtk_lat"], frame["rtk_lon"]))

        # RTK yaw (rtk_heading is already in the project yaw convention: 0=E, 90=N)
        yaw = self.converter.rtk_heading_to_yaw(frame["rtk_heading"])

        # Apply extrinsic: RTK -> camera. Offsets are given in robot/body frame
        # and must be rotated into world frame using yaw.
        dx_b = float(RTK_TO_CAMERA_OFFSET_X_FWD_M)
        dy_b = float(RTK_TO_CAMERA_OFFSET_Y_LEFT_M)
        c = math.cos(math.radians(yaw))
        s = math.sin(math.radians(yaw))
        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        # Convert RTK position to camera position
        x = x - dx_w
        y = y - dy_w

        # Store lat/lon for later error computation
        self.correct_pose_latlon = (float(frame["rtk_lat"]), float(frame["rtk_lon"]))
        self.gps_pose_latlon = (float(frame["gps_lat"]), float(frame["gps_lon"]))
        self._last_rtk_yaw_deg = float(yaw)

        return Pose2d(x=x, y=y, yaw=yaw)

    def get_next_image(
        self,
    ) -> tuple[str, list[Point], Pose2d | None]:
        """Retrieves the next image, shows it for manual tree picking, and
        returns the clicks and pose.

        :return:
          - image_name: filename or "0" when exhausted
          - selected_points: list of image-pixel Points
          - pose: The true corresponding Pose2d from the RTK or None
        """
        if self._current_index >= len(self._image_list):
            return "0", [], None

        # Grab and advance index (random or sequential)
        image_name = self._image_list[self._current_index]
        if self.randomize:
            self._current_index = random.randint(0, len(self._image_list) - 1)
        else:
            self._current_index += 1

        # Get the pose for this image
        pose = self.get_current_pose(image_name)
        selected_points: list[Point] = []

        if not pose:
            return image_name, selected_points, None

        # Load and display for click events
        path = os.path.join(self.image_folder, image_name)
        image = cv2.imread(path)
        image.copy()
        if image is None:
            print(f"Error loading image: {image_name}")
            return self.get_next_image()

        # Get image dimensions
        img_h, img_w = image.shape[:2]

        # Create a copy to display and modify
        display_max_w = 1280
        display_max_h = 720
        scale = min(1.0, display_max_w / img_w, display_max_h / img_h)
        if scale < 1.0:
            display_w = int(round(img_w * scale))
            display_h = int(round(img_h * scale))
            displayed_image = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        else:
            display_w = img_w
            display_h = img_h
            displayed_image = image.copy()

        # Create a window name with image info for uniqueness
        window_name = f"Select Points - {image_name}"

        # Create window and set it to a fixed size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_w, display_h)
        cv2.imshow(window_name, displayed_image)

        # Define the mouse callback function
        def click_event(event: int, x: int, y: int, flags: int, param: Any | None) -> None:
            """Handle mouse click events for point selection."""
            if event == cv2.EVENT_LBUTTONDOWN:
                full_x = int(round(x / scale)) if scale != 1.0 else x
                full_y = int(round(y / scale)) if scale != 1.0 else y
                selected_points.append(Point(full_x, full_y))
                cv2.circle(displayed_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(window_name, displayed_image)

        # Set the mouse callback function
        cv2.setMouseCallback(window_name, click_event)  # type: ignore[arg-type]

        print(f"Processing image: {image_name}")
        print("Left-click to select trees, press Enter when done, ESC to skip")

        # Wait for keypress - Enter key will finish selection
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if not selected_points:
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                    return image_name, [], None
                break
            if key == 27:  # ESC
                selected_points = []
                break

        # Close the specific window
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)

        return image_name, selected_points, pose


    def run(self) -> None:
        """
        Main application loop: for each image, collect clicks,
        compute ground angles, match trees, and print errors.
        """

        # Prints out the instructions for use
        print("Starting manual tree selection...")
        print("Instructions:")
        print("- Left-click to select trees in the image")
        print("- Press Enter when done selecting trees")
        print("- Press ESC to skip current image")
        print("- Press Ctrl+C to quit")
        print("- Debug plots will be saved to 'debug_plots/' folder\n")

        DebugVisualizer.clear_plots()

        while True:
            # Gets the name, points chosen, and the pose of the next image
            name, points, pose = self.get_next_image()
            if name == "0":
                print("No more images. Exiting.")
                return

            if pose is None:
                print(f"Skipping {name}: missing/invalid RTK heading")
                continue

            # Load image
            path = os.path.join(self.image_folder, name)
            image = cv2.imread(path)
            if image is None:
                print(f"Error loading image: {name}")
                continue

            # Gets the ground thetas from the image and matches the corresponding trees with the
            # satellite data
            img_h, img_w = image.shape[:2]
            ground_thetas = [self.converter.image_x_to_theta(pt.x, img_w) for pt in points]

            if PLOT:
                save_name = os.path.splitext(name)[0]
                aoi_trees = self.tree_matcher._get_area_of_interest(pose)
                DebugVisualizer.plot_thetas(
                    ground_thetas,
                    pose,
                    aoi_trees,
                    f"{save_name}_thetas",
                )

            est_xy = self.tree_matcher.match_trees(pose, ground_thetas)

            print("Estimated location xy: ", est_xy)

            # Create debug visualization (saved to file, no display conflicts)
            if PLOT:
                # Use image name (without extension) as save name
                save_name = os.path.splitext(name)[0]
                DebugVisualizer.plot_aoi(
                    self.tree_matcher.satellite_tree_locations,
                    self.tree_matcher.aoi_trees,
                    pose,
                    save_name,
                )

            # Print out GPS vs SGIL errors (in local XY meters)
            assert self.correct_pose_latlon and self.gps_pose_latlon, "Pose info missing!"

            # RTK (camera) XY is `pose.x, pose.y` already.
            rtk_cam_xy = Point(pose.x, pose.y)

            # Convert GPS lat/lon to XY (GPS antenna), then apply the same RTK->camera offset
            gps_x, gps_y = self.converter.latlon_to_xy(self.gps_pose_latlon)

            # Rotate the body offset into world frame using RTK yaw
            dx_b = float(RTK_TO_CAMERA_OFFSET_X_FWD_M)
            dy_b = float(RTK_TO_CAMERA_OFFSET_Y_LEFT_M)
            c = math.cos(math.radians(pose.yaw))
            s = math.sin(math.radians(pose.yaw))
            dx_w = c * dx_b - s * dy_b
            dy_w = s * dx_b + c * dy_b

            gps_cam_xy = Point(gps_x - dx_w, gps_y - dy_w)

            # Euclidean errors in meters
            gps_err = math.hypot(gps_cam_xy.x - rtk_cam_xy.x, gps_cam_xy.y - rtk_cam_xy.y)
            sgil_err = math.hypot(est_xy.x - rtk_cam_xy.x, est_xy.y - rtk_cam_xy.y)

            logging.info(f"Image: {name}")
            logging.info(f"  Estimated XY:      ({est_xy.x:.2f}, {est_xy.y:.2f})")
            logging.info(f"  GPS error (m):     {gps_err:.2f}")
            logging.info(f"  SGIL error (m):    {sgil_err:.2f}")
            logging.info(f"  RTK Yaw (deg):     {pose.yaw:.2f}")

            # If you still want lat/lon for display:
            # est_latlon = self.converter.xy_to_latlon(est_xy)

            # Done with this image


if __name__ == "__main__":
    ManualSGIL().run()
