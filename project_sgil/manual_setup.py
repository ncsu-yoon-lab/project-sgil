"""
This program is intended to allow a user to manually select the trees from
the ground view and used pre-collected and pre-labeled data. This can be
swapped out with an automatic version once the matching is completed.
"""

# Import the necessary libraries
import os
import random
import cv2
import pandas as pd
from constants import *
from converter import Converter
from tree_matcher import TreeMatcher

class ManualSelector():

    def __init__(self):
        self.converter = Converter(ORIGIN[0], ORIGIN[1])
        self.tree_matcher = TreeMatcher()

        self.robot_data_log = pd.read_csv(DATA_LOGGER_PATH)

        self.correct_pose = None
        self.gps_pose = None
        
        # Initialize image list and current index as instance variables
        self.image_list = [
            f for f in sorted(os.listdir(IMAGE_FOLDER_PATH)) if f.lower().endswith(".jpg")
        ]
        self.current_index = 0

    def get_current_pose(self, image_name):
        matching_rows = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]

        # Check if we found any matching rows
        if matching_rows.empty:
            return None

        # Get the first matching row
        matching_row = matching_rows.iloc[0]

        if pd.isna(matching_row["rtk_heading"]):
            return None

        gps_pose = self.get_gps_pose(matching_row)

        return gps_pose

    def get_gps_pose(self, row):
        xy_point = self.converter.latlon_to_xy((row["rtk_lat"], row["rtk_lon"]))

        # Taking heading from rtk for now, need to fix eventually
        yaw_deg = self.converter.heading_to_yaw(row["rtk_heading"])

        self.correct_pose = (row["rtk_lat"], row["rtk_lon"])
        self.gps_pose = (row["gps_lat"], row["gps_lon"])

        return (xy_point[0], xy_point[1], yaw_deg)

    def get_next_image(self):
        # If all images are processed, return '0' and an empty list
        if self.current_index >= len(self.image_list):
            return "0", [], None

        # Get the current image name and increment the index
        image_name = self.image_list[self.current_index]

        if RANDOM:
            self.current_index = random.randint(1, len(self.image_list) - 1)
        else:
            self.current_index += 1

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

    def reset_image_iteration(self):
        """Reset the image iteration to start from the beginning"""
        self.current_index = 0

    def main(self) -> None:
        while True:
            image_name, points, current_pose = self.get_next_image()

            if image_name == "0":
                print("No more images.")
                break

            ground_thetas = []

            for point in points:
                ground_thetas.append(self.converter.image_x_to_theta(point[0]))

            estimated_location_xy = self.tree_matcher.match_trees(current_pose, ground_thetas)

            estimated_location_latlon = self.converter.xy_to_latlon(estimated_location_xy)

            print(estimated_location_latlon)
            print(
                f"GPS Error: {self.converter.haversine(self.gps_pose[0], self.gps_pose[1], self.correct_pose[0], self.correct_pose[1])}"
            )
            print(
                f"SGIL Error: {self.converter.haversine(estimated_location_latlon[0], estimated_location_latlon[1], self.correct_pose[0], self.correct_pose[1])}"
            )


if __name__ == "__main__":
    selector = ManualSelector()

    try:
        selector.main()
    except KeyboardInterrupt as e:
        quit()