"""
This program is intended to allow a user to manually select the trees from the ground view and
used pre-collected and pre-labeled data.
"""

# Import the necessary libraries
import os

import cv2
import matplotlib.pyplot as plt
from constants import *
from converter import Converter
from tree_matcher import TreeMatcher

converter = Converter(ORIGIN[0], ORIGIN[1])

tree_matcher = TreeMatcher()


def plot_points(points) -> None:
    x = []
    y = []

    for point in points:
        x.append(point[0])
        y.append(point[1])

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=2, c="g", vmin=0, vmax=100)

    plt.show()


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
        return "0", []

    # Get the current image name and increment the index
    image_name = get_next_image.image_list[get_next_image.current_index]
    get_next_image.current_index += 1

    # Load the image
    image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
    image = cv2.imread(image_path)

    # List to store selected points
    selected_points = []

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
            # print(f"Point added: ({x}, {y})")

    # Create a copy to display and modify
    displayed_image = image.copy()

    # Create a window and set the callback function
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", click_event)

    # Display initial image
    cv2.imshow("Select Points", displayed_image)
    # print(f"Image: {image_name}")
    # print("Click on the image to select points. Press Enter when done.")

    # Wait for keypress - Enter key will finish selection
    while True:
        key = cv2.waitKey(1) & 0xFF
        # If Enter key is pressed, break the loop
        if key == 13:  # 13 is the ASCII code for Enter
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # print(f"Selected {len(selected_points)} points")

    return image_name, selected_points


def main() -> None:
    # Initialize the list of trees
    sat_tree_locations = []

    if PLOT:
        plot_points(sat_tree_locations)

    while True:
        image_name, points = get_next_image()
        ground_thetas = []
        if image_name == "0":
            print("No more images.")
            break

        for point in points:
            ground_thetas.append(converter.image_x_to_theta(point[0]))


if __name__ == "__main__":
    main()
