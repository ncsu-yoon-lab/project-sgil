'''
This program is intended to allow a user to manually select the trees from the ground view and 
used pre-collected and pre-labeled data
'''

# Import the necessary libraries
import os
import cv2
import csv
import math
import matplotlib.pyplot as plt
from constants import *
from converter import Converter
from tree_matcher import TreeMatcher
import pandas as pd

converter = Converter(ORIGIN[0], ORIGIN[1])

tree_matcher = TreeMatcher()

# Open the data log csv
robot_data_log = pd.read_csv(DATA_LOGGER_PATH)

# Correct location
correct_pose = None
gps_pose = None

def get_current_pose(image_name):

    matching_rows = robot_data_log[robot_data_log['image_name'].str.contains(image_name, case=False, na=False)]
    
    # Check if we found any matching rows
    if matching_rows.empty:
        return None
        
    # Get the first matching row
    matching_row = matching_rows.iloc[0]

    if pd.isna(matching_row['gps_track']):
        return None
    
    gps_pose = get_gps_pose(matching_row)

    return gps_pose

def get_gps_pose(row):
    global correct_pose, gps_pose
    xy_point = converter.latlon_to_xy((row['gps_latitude'], row['gps_longitude']))

    yaw_deg = converter.heading_to_yaw(row['gps_track'])

    correct_pose = (row['swift_latitude'], row['swift_longitude'])
    gps_pose = (row['gps_latitude'], row['gps_longitude'])

    # Backup the position based on the assumed error of the GPS
    reverse_yaw_rad = math.radians((yaw_deg - 180) % 360)
    xy_point = (math.cos(reverse_yaw_rad) * GPS_ERROR_M + xy_point[0], math.sin(reverse_yaw_rad) * GPS_ERROR_M + xy_point[1])

    return (xy_point[0], xy_point[1], yaw_deg)

def get_next_image():
    # Define static variables to keep track of state
    if not hasattr(get_next_image, "image_list"):
        # Initialize the list of .jpg images and the index
        get_next_image.image_list = [
            f for f in sorted(os.listdir(IMAGE_FOLDER_PATH)) if f.lower().endswith(".jpg")
        ]
        get_next_image.current_index = 0

    # If all images are processed, return '0' and an empty list
    if get_next_image.current_index >= len(get_next_image.image_list):
        return '0', [], None

    # Get the current image name and increment the index
    image_name = get_next_image.image_list[get_next_image.current_index]

    get_next_image.current_index += 1

    pose = get_current_pose(image_name)

    # List to store selected points
    selected_points = []
    
    if pose is not None:
        # Load the image
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
        image = cv2.imread(image_path)
        
        # Define the mouse callback function
        def click_event(event, x, y, flags, param):
            # Check if left mouse button was clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point to list
                selected_points.append((x, y))
                # Draw circle at clicked position
                cv2.circle(displayed_image, (x, y), 5, (0, 255, 0), -1)
                # Update the display
                cv2.imshow("Select Points", displayed_image)
                # print(f"Point added: ({x}, {y})")
        
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
                break
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
    
    # print(f"Selected {len(selected_points)} points")
    
    return image_name, selected_points, pose

def main():
    # Initialize the list of trees
    sat_tree_locations = []

    while True:
        image_name, points, current_pose = get_next_image()
    
        if image_name == '0':
            print("No more images.")
            break

        if current_pose is not None:

            ground_thetas = []

            for point in points:
                ground_thetas.append(converter.image_x_to_theta(point[0]))
                
            estimated_location_xy = tree_matcher.match_trees(current_pose, ground_thetas)

            estimated_location_latlon = converter.xy_to_latlon(estimated_location_xy)

            print(estimated_location_latlon)
            print(f"GPS Error: {converter.haversine(gps_pose[0], gps_pose[1], correct_pose[0], correct_pose[1])}")
            print(f"SGIL Error: {converter.haversine(estimated_location_latlon[0], estimated_location_latlon[1], correct_pose[0], correct_pose[1])}")


if __name__ == "__main__":
    main()
