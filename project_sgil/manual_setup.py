import logging
import os
import random

import cv2
import pandas as pd
from constants import DATA_LOGGER_PATH, IMAGE_FOLDER_PATH, ORIGIN, RANDOM
from converter import Converter
from data_structs import Point, Pose2d
from tree_matcher import TreeMatcher

logging.basicConfig(level=logging.INFO)


class SGILMatcherApp:
    """
    Encapsulates the SGIL tree‐matching workflow:
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
        :param data_log_path: CSV with columns including
                              image_filename, rtk_lat, rtk_lon,
                              rtk_heading, gps_lat, gps_lon.
        :param origin: (lat, lon) of the converter origin.
        :param randomize: Whether to pick the next image at random.
        """
        self.converter = Converter(origin[0], origin[1])
        self.tree_matcher = TreeMatcher()
        self.image_folder = image_folder
        self.randomize = randomize

        # Load robot log into a DataFrame once
        self.robot_data_log = pd.read_csv(data_log_path)

        # Will be set on each call to get_gps_pose
        self.correct_pose: tuple[float, float] | None = None
        self.gps_pose: tuple[float, float] | None = None

        # Prepare image list state
        self._image_list: list[str] = sorted(
            f for f in os.listdir(self.image_folder) if f.lower().endswith(".jpg")
        )
        self._current_index: int = 0

    def get_current_pose(self, image_name: str) -> Pose2d | None:
        """
        Lookup the RTK/GPS pose for a given image filename.

        :param image_name: Name of the .jpg image.
        :return: Pose2d if RTK heading is valid; otherwise None.
        """
        df = self.robot_data_log
        matches = df[df["image_filename"].str.contains(image_name, case=False, na=False)]
        if matches.empty or pd.isna(matches.iloc[0]["rtk_heading"]):
            return None

        return self.get_gps_pose(matches.iloc[0])

    def get_gps_pose(self, row: pd.Series) -> Pose2d:
        """
        Convert a log row into a Pose2d using RTK for yaw.

        Also updates internal gps_pose/correct_pose for error logging.
        :param row: pandas Series with rtk_lat, rtk_lon, rtk_heading,
            gps_lat, gps_lon.
        :return: Pose2d in local XY + yaw degrees.
        """
        # Convert lat/lon to XY
        x, y = self.converter.latlon_to_xy((row["rtk_lat"], row["rtk_lon"]))
        yaw = self.converter.heading_to_yaw(row["rtk_heading"])

        # Store lat/lon for later error computation
        self.correct_pose = (row["rtk_lat"], row["rtk_lon"])
        self.gps_pose = (row["gps_lat"], row["gps_lon"])

        return Pose2d(x=x, y=y, yaw=yaw)

    def get_next_image(
        self,
    ) -> tuple[str, list[Point], Pose2d | None]:
        """
        Retrieves the next image, shows it for manual tree picking, and
        returns the clicks and pose.

        :return:
          - image_name: filename or "0" when exhausted
          - selected_points: list of image‐pixel Points
          - pose: corresponding Pose2d or None
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
        display = image.copy()

        def click_event(evt, x, y, flags, param) -> None:
            if evt == cv2.EVENT_LBUTTONDOWN:
                selected_points.append(Point(x, y))
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Trees", display)

        cv2.namedWindow("Select Trees")
        cv2.setMouseCallback("Select Trees", click_event)
        cv2.imshow("Select Trees", display)

        # Wait until Enter is pressed
        while True:
            if (cv2.waitKey(1) & 0xFF) == 13:
                break
        cv2.destroyAllWindows()

        return image_name, selected_points, pose

    def run(self) -> None:
        """
        Main application loop: for each image, collect clicks,
        compute ground angles, match trees, and print errors.
        """
        while True:
            name, points, pose = self.get_next_image()
            if name == "0":
                logging.info("All images processed. Exiting.")
                break

            if not pose:
                logging.warning(f"No valid pose for {name}; skipping.")
                continue

            ground_thetas = [self.converter.image_x_to_theta(pt.x) for pt in points]
            est_xy = self.tree_matcher.match_trees(pose, ground_thetas)

            # Convert back to lat/lon
            est_latlon = self.converter.xy_to_latlon(est_xy)

            # Print out GPS vs SGIL errors
            assert self.correct_pose and self.gps_pose, "Pose info missing!"
            gps_err = self.converter.haversine(
                self.gps_pose[0],
                self.gps_pose[1],
                self.correct_pose[0],
                self.correct_pose[1],
            )
            sgil_err = self.converter.haversine(
                est_latlon[0],
                est_latlon[1],
                self.correct_pose[0],
                self.correct_pose[1],
            )

            logging.info(f"Image: {name}")
            logging.info(f"  Estimated LatLon: {est_latlon}")
            logging.info(f"  GPS error (m):     {gps_err:.2f}")
            logging.info(f"  SGIL error (m):    {sgil_err:.2f}")


if __name__ == "__main__":
    SGILMatcherApp().run()
