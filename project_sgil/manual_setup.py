"""
Manual tree selection interface for ground view analysis and matching with
satellite data using pre-collected and pre-labeled dataset.

file: manual_selector.py
author: Cole Malinchock and Jack Elia
"""

# Import the necessary libraries
import os
import random
import cv2
import pandas as pd

# Import custom classes
from constants import *
from converter import Converter
from data_structs import *
from tree_matcher import TreeMatcher


class ManualSelector:
    """
    Manual tree selector for selecting trees from pre-collected campus
    imagery. Designed to be replaced by automated detection system.
    """

    def __init__(self) -> None:
        """
        Initialize the ManualSelector with converter, tree matcher, and data
        logging capabilities.
        """
        
        # Creates the converter and tree matcher
        self.converter = Converter(ORIGIN[0], ORIGIN[1])
        self.tree_matcher = TreeMatcher()

        # Opens the csv on the data collected by the robot
        self.robot_data_log = pd.read_csv(DATA_LOGGER_PATH)

        # Initializes the gps pose and correct pose
        self.correct_pose = None
        self.gps_pose = None
        
        # Initialize image list and current index as instance variables
        self.image_list = [
            f for f in sorted(os.listdir(IMAGE_FOLDER_PATH)) if f.lower().endswith(".jpg")
        ]
        self.current_index = 0

    def get_current_pose(self, image_name: str) -> Pose2d:
        """
        Extract pose information from data log based on image filename.

        :param image_name: Image filename to match against log entries.
        :return: GPS pose as (x, y, heading) or None if not found.
        """

        # Gets the row from the log with the image filename
        matching_rows = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]

        # Check if we found any matching rows
        if matching_rows.empty:
            return None

        # Get the first matching row
        matching_row = matching_rows.iloc[0]

        # Checks if the heading is none and returns None
        if pd.isna(matching_row["rtk_heading"]):
            return None

        # Gets the gps pose from the matching row
        gps_pose = self.get_gps_pose(matching_row)

        return gps_pose

    def get_gps_pose(self, row) -> Pose2d:
        """
        Convert GPS coordinates from data log row to local coordinate system.

        :param row: DataFrame row containing RTK GPS data.
        :return: Pose in local coordinates as (x, y, yaw).
        """

        # Gets the x, y point by converting the values of the rows lat, lon
        xy_point = self.converter.latlon_to_xy((row["rtk_lat"], row["rtk_lon"]))

        # Taking heading from rtk for now, need to fix eventually
        yaw_deg = self.converter.heading_to_yaw(row["rtk_heading"])

        # Gets the correct pose from the rtk
        self.correct_pose = (row["rtk_lat"], row["rtk_lon"])

        return Pose2d(xy_point[0], xy_point[1], yaw_deg)

    def get_next_image(self) -> tuple[str, list[float], Pose2d]:
        """
        Load next image for manual tree selection with interactive interface.

        :return: Tuple of (image_name, selected_points, pose) or ("0", [], None)
            if no more images.
        """

        # If all images are processed, return '0' and an empty list
        if self.current_index >= len(self.image_list):
            return "0", [], None

        # Get the current image name and increment the index
        image_name = self.image_list[self.current_index]

        # If randomly generating images or not
        if RANDOM:
            self.current_index = random.randint(1, len(self.image_list) - 1)
        else:
            self.current_index += 1

        # Gets the pose from the current image name
        pose = self.get_current_pose(image_name)

        # If pose is None, recursively call to get the next image
        if pose is None:
            return self.get_next_image()

        # List to store selected points
        selected_points = []

        # Load the image
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
        image = cv2.imread(image_path)

        # Define the mouse callback function
        def click_event(event, x, y, flags, param) -> None:
            """Handle mouse click events for point selection."""
            # Check if left mouse button was clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point to list
                selected_points.append((x, y))
                # Draw circle at clicked position
                cv2.circle(displayed_image, (x, y), 5, (0, 255, 0), -1)
                # Update the display
                cv2.imshow("Select Points", displayed_image)

        # Create a copy to display and modify
        displayed_image = image.copy()

        # Create a window and set the callback function
        cv2.namedWindow("Select Points")
        cv2.setMouseCallback("Select Points", click_event)

        # Display initial image
        cv2.imshow("Select Points", displayed_image)

        # Wait for keypress - Enter key will finish selection
        while True:
            key = cv2.waitKey(1) & 0xFF
            # If Enter key is pressed, break the loop
            if key == 13:  # 13 is the ASCII code for Enter
                cv2.destroyAllWindows()
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        return image_name, selected_points, pose

    def reset_image_iteration(self) -> None:
        """
        Reset image iteration counter to start processing from beginning.
        """

        # Set the current index to 0
        self.current_index = 0

    def main(self) -> None:
        """
        Main execution loop for manual tree selection and position estimation.
        """

        # Loops until a break
        while True:

            # Gets the next image
            image_name, points, current_pose = self.get_next_image()

            # Checks if there are no more images
            if image_name == "0":
                print("No more images.")
                break
                
            # Initializes a list of ground thetas
            ground_thetas = []

            # Loops through all points from the image and appends them to the ground thetas
            for point in points:
                ground_thetas.append(self.converter.image_x_to_theta(point[0]))

            # Gets the estimated x, y position from tree matcher
            estimated_location_xy = self.tree_matcher.match_trees(current_pose, ground_thetas)

            # Converts the x, y to lat, lon
            estimated_location_latlon = self.converter.xy_to_latlon(estimated_location_xy)

            # Prints the results
            print(estimated_location_latlon)
            print(
                f"SGIL Error: {self.converter.haversine(estimated_location_latlon[0], estimated_location_latlon[1], self.correct_pose[0], self.correct_pose[1])}"
            )


if __name__ == "__main__":

    # Creates the manual selector object
    selector = ManualSelector()

    # Tries the main until there is a keyboard interrupt
    try:
        selector.main()
    except KeyboardInterrupt as e:
        quit()