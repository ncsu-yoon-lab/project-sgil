"""
Manual tree selection interface for ground view analysis and matching with satellite data using pre-
collected and pre-labeled dataset.

file: manual_selector.py
author: Cole Malinchock and Jack Elia
"""

# Import standard libraries
import logging
import os
import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Import custom classes
from constants import (
    CAMERA_HEIGHT_M,
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
    RANDOM,
)
from converter import Converter
from data_structs import Point, Pose2d
from debug_visualizer import DebugVisualizer
from path_vector_generator import PathVectorGenerator
from tree_matcher import TreeMatcher

logging.basicConfig(level=logging.INFO)


class SGILMatcherApp:
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
        self.image_shape = IMAGE_SHAPE
        self.camera_height_m = CAMERA_HEIGHT_M

        # Load robot log into a DataFrame once
        self.robot_data_log = pd.read_csv(data_log_path)
        self.path_vector_generator = PathVectorGenerator(
            image_shape=self.image_shape,
            camera_height=self.camera_height_m,
            dataset=self.robot_data_log,
        )

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
        :param row: pandas Series with rtk_lat, rtk_lon, rtk_heading, gps_lat, gps_lon.
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
        Retrieves the next image, shows it for manual tree picking, and returns the clicks and pose.

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

        # Performs the vectorization of the path detection
        path_vector, path_yaw = self.path_vector_generator.get_path_vector_and_yaw(
            image, self._current_index
        )
        print(
            f"""Vector:
            <({path_vector[0][0]}, {path_vector[0][1]}),
            ({path_vector[1][0]}, {path_vector[1][1]})>"""
        )
        print(f"Yaw: {path_yaw} deg")

        # Define the mouse callback function
        def click_event(event: int, x: int, y: int, flags: list, param: any) -> None:
            """
            Handle mouse click events for point selection.
            """
            # Check if left mouse button was clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point to list
                selected_points.append(Point(x, y))
                # Draw circle at clicked position
                cv2.circle(displayed_image, (x, y), 5, (0, 255, 0), -1)
                # Update the display
                cv2.imshow(window_name, displayed_image)

        # Close any existing matplotlib figures and OpenCV windows
        plt.close("all")
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Create a copy to display and modify
        displayed_image = image.copy()

        # Create a window name with image info for uniqueness
        window_name = f"Select Points - {image_name}"

        # Create window and set it to autosize first, then resize
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, displayed_image)

        # Set the mouse callback function
        cv2.setMouseCallback(window_name, click_event)

        print(f"Processing image: {image_name}")
        print("Left-click to select trees, press Enter when done, ESC to skip")

        # Wait for keypress - Enter key will finish selection
        while True:
            key = cv2.waitKey(1) & 0xFF
            # If Enter key is pressed, break the loop
            if key == 13:  # 13 is the ASCII code for Enter
                break
            # If ESC key is pressed, skip this image
            elif key == 27:  # 27 is the ASCII code for ESC
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

        # Continues until there are no more images
        while True:
            # Gets the name, points chosen, and the pose of the next image
            name, points, pose = self.get_next_image()

            # Checks that there is another image and a pose
            if name == "0":
                logging.info("All images processed. Exiting.")
                break

            if not pose:
                logging.warning(f"No valid pose for {name}; skipping.")
                continue

            # Gets the ground thetas from the image and matches the corresponding trees with the
            # satellite data
            ground_thetas = [self.converter.image_x_to_theta(pt.x) for pt in points]
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
                # DebugVisualizer.plot_wedges(self.tree_matcher.wedges, pose,
                # self.tree_matcher.aoi_trees, est_xy, save_name)

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

            # TODO: remove this break to process all images
            break


if __name__ == "__main__":
    SGILMatcherApp().run()
