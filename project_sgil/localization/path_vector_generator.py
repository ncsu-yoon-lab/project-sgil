"""
Manual path vector selection that is capable of producing a path vector from an image.

file: path_vector.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
import math
import sys

import cv2
import numpy as np
import pandas as pd
from OSMPythonTools.overpass import Overpass

from project_sgil.constants import H_FOV_DEG, ORIGIN, V_FOV_DEG
from project_sgil.data_structs import Point

# Import custom libraries
from project_sgil.utils.converter import Converter


class PathVectorGenerator:
    """
    Encapsulates the Path Vector Generator:
      1. Read image and manually select edges of the path in the image if it exists
      2. Uses OSM to find the path width and global nodes of the path
      3. Uses the vanishing point of the lines detected to find the heading and
         displacement relative to the path
      4. Outputs a global vector of where the vehicle lies on the path and the
         estimated heading from the path
    """

    def __init__(
        self, image_shape: tuple[int, int, int], camera_height: float, dataset: pd.DataFrame
    ) -> None:
        """
        :param image_shape: The shape of the images provided
        :param camera_height: The height of the camera
        :param dataset: The working dataset of the project
        """
        self.height, self.width, _ = image_shape
        self.points = None
        self.fx = (self.width / 2) / math.tan(math.radians(H_FOV_DEG / 2))
        self.fy = (self.height / 2) / math.tan(math.radians(V_FOV_DEG / 2))
        self.center_point = Point(x=self.width / 2, y=self.height / 2)
        self.camera_height = camera_height
        self.dataset = dataset
        self.overpass = Overpass()
        self.converter = Converter(ORIGIN[0], ORIGIN[1])

    def click_event(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: tuple[list[tuple[int, int]], np.ndarray],
    ) -> None:
        """
        Handles the click events.

        :param event: The click event instance occuring
        :param x: The x location of the event
        :param y: The y location of the event
        :param flags: The flags of the event
        :param param: The parameters of the event
        """

        # Gets the parameters from the click event
        img_display = param

        # Adds a circle at the location of the click
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append(Point(x=x, y=y))
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)  # draw green dot
            cv2.putText(
                img_display, f"P{len(self.points)}", (x, y), 0, 1.0, (0, 255, 0), thickness=2
            )
            cv2.imshow("Select Path Edges", img_display)

    def get_intersection(
        self,
        left_points: tuple[Point, Point],
        right_points: tuple[Point, Point],
    ) -> tuple[Point, float, float]:
        """
        Gets the intersection between the left and right points and the corresponding slopes.

        :param left_points: The left two points of the edge detected
        :param right_points: The right two points of the edge detected
        :return: The intersection point and the slope of the left and right lines
        """

        # Gets the slope of the left and right points
        if left_points[0].x == left_points[1].x:
            m_left = 9999.0
        else:
            m_left = (left_points[0].y - left_points[1].y) / (left_points[0].x - left_points[1].x)
        if right_points[0].x == right_points[1].x:
            m_right = 9999.0
        else:
            m_right = (right_points[0].y - right_points[1].y) / (
                right_points[0].x - right_points[1].x
            )

        # Gets the y-intercept of the left and right lines
        b_left = left_points[0].y - m_left * left_points[0].x
        b_right = right_points[0].y - m_right * right_points[0].x

        # Gets the x and y intersection point
        x_int = (b_left - b_right) / (m_right - m_left)
        y_int = m_left * x_int + b_left

        return Point(x=x_int, y=y_int), m_left, m_right

    def get_displacement_and_yaw(
        self,
        intersection_point: Point,
        m_left: float,
        m_right: float,
        path_width: float,
        path_yaw: float,
    ) -> tuple[float, float]:
        """
        Gets the displacement and yaw of the system based on the results of the edges detected.

        :param intersection_point: The intersecting point between the two edges
        :param m_left: The slope of the left line
        :param m_right: The slope of the right line
        :param path_width: The width of the path
        :param path_yaw: The yaw of the path globally
        :return: The estimated displacement and yaw of the system relative to the path
        """

        # Finds the difference in the x of the center point and intersection
        dx = intersection_point.x - self.center_point.x

        # Gets the relative yaw based on the difference in x and the focal in the x
        relative_yaw_deg = math.degrees(math.atan(dx / self.fx))

        global_yaw = path_yaw - relative_yaw_deg
        global_yaw = global_yaw % 360

        x_delta = dx * (self.camera_height / self.fy)

        return x_delta, global_yaw

    def get_closest_point_along_line(self, p1: Point, p2: Point, p3: Point) -> float:
        """
        Find the point where p3 intersects perpendicular to the line from p1 to p2.

        :param p1: The first point of the line
        :param p2: The second point of the line
        :param p3: The third point to find the distance to the line
        :return: The distance from p1 to the line from p1 to p2
        """

        # Get the inverse slope of the line
        m = 9999 if p1.x == p2.x else (p2.y - p1.y) / (p2.x - p1.x)
        m_inv = -1 / m

        # Get the inverse y intercept of the line
        b = p1.y - m * p1.x
        b_inv = p3.y - m_inv * p3.x

        # Find the intersection point between the line and p3
        x_int = (b - b_inv) / (m_inv - m)
        y_int = x_int * m + b

        # Calculate its distance
        dist = math.sqrt((x_int - p3.x) ** 2 + (y_int - p3.y) ** 2)

        return dist

    def get_path_data(
        self,
        estimated_position_latlon: tuple[float, float],
        estimated_position_xy: Point,
        estimated_heading_deg: float,
    ) -> tuple[Point, Point, float, float]:
        """
        Gets the data from the path to be handled.

        :param estimated_position_latlon: The estimated lat lon of the system
        :param estimated_position_xy: The estimated x y of the system
        :param estimated_heading_deg: The estimated heading of the system
        :return: The start and end points of the path and the width of the path
        """

        # Extracts the lat and lon from the estimation
        lat, lon = estimated_position_latlon

        # Creates a query to request from OSM
        query = f"""way(around:50,{lat},{lon})[highway~"footway|path|pedestrian"];out geom;"""
        result = self.overpass.query(query)

        # Loops through all the ways gathered from the query
        closest_dist = float("inf")
        for way in result.elements():
            # Loops through all the nodes of each way
            previous_node = None
            for node in way.geometry()["coordinates"]:
                latlon_node = (node[1], node[0])
                # Checks that there is a previous node
                if previous_node is not None:
                    # Gets the distance from estimated position to the path nodes

                    prev_node_xy = Point(*self.converter.latlon_to_xy(previous_node))
                    cur_node_xy = Point(*self.converter.latlon_to_xy(latlon_node))

                    dist = self.get_closest_point_along_line(
                        prev_node_xy, cur_node_xy, estimated_position_xy
                    )

                    # Checks if the distance is the new shortest distance
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_path_width = float(way.tags().get("width"))
                        closest_prev_node_xy = prev_node_xy
                        closest_cur_node_xy = cur_node_xy

                    previous_node = latlon_node
                else:
                    previous_node = latlon_node

        # Gets the delta to each node from estimated pose
        estimated_pose_vector = Point(
            x=math.cos(math.radians(estimated_heading_deg)),
            y=math.sin(math.radians(estimated_heading_deg)),
        )
        prev_node_vector = Point(
            x=closest_prev_node_xy.x - estimated_position_xy.x,
            y=closest_prev_node_xy.y - estimated_position_xy.y,
        )
        prev_node_vector_dist = math.sqrt(prev_node_vector.x**2 + prev_node_vector.y**2)
        cur_node_vector = Point(
            x=closest_cur_node_xy.x - estimated_position_xy.x,
            y=closest_cur_node_xy.y - estimated_position_xy.y,
        )
        cur_node_vector_dist = math.sqrt(cur_node_vector.x**2 + cur_node_vector.y**2)
        delta_heading_prev_node = math.acos(
            (
                estimated_pose_vector.x * prev_node_vector.x
                + estimated_pose_vector.y * prev_node_vector.y
            )
            / prev_node_vector_dist
        )
        delta_heading_cur_node = math.acos(
            (
                estimated_pose_vector.x * cur_node_vector.x
                + estimated_pose_vector.y * cur_node_vector.y
            )
            / cur_node_vector_dist
        )

        # Checks if heading is closer to the current node or previous node to identify
        # the start and end
        if delta_heading_prev_node > delta_heading_cur_node:
            start_node = closest_cur_node_xy
            end_node = closest_prev_node_xy

        else:
            start_node = closest_prev_node_xy
            end_node = closest_cur_node_xy

        # Get the delta in the start and end nodes to get the heading of the path
        dx = end_node.x - start_node.x
        dy = end_node.y - start_node.y
        closest_path_heading = (90 - math.degrees(math.atan2(dy, dx))) % 360

        return start_node, end_node, closest_path_width, closest_path_heading

    def get_path_vector(
        self,
        start_node_xy: Point,
        end_node_xy: Point,
        displacement: float,
    ) -> tuple[Point, Point]:
        """
        Gets the vector of the path from the node and displacement.

        :param start_node_xy: The starting point of the path
        :param end_node_xy: The ending point of the path
        :param displacement: The displacement on the path
        :return: The start and end points from the displacement perpendicular to the path
        """

        # Creates a unit vector of the path vector
        path_dist = math.sqrt(
            (end_node_xy.x - start_node_xy.x) ** 2 + (end_node_xy.y - start_node_xy.y) ** 2
        )
        path_unit_vector = Point(
            x=(end_node_xy.x - start_node_xy.x) / path_dist,
            y=(end_node_xy.y - start_node_xy.y) / path_dist,
        )

        # Creates the inverted unit vector of the path based on the direction of the displacement
        if displacement < 0:
            inv_path_unit_vector = Point(x=path_unit_vector.y, y=-path_unit_vector.x)
        else:
            inv_path_unit_vector = Point(x=-path_unit_vector.y, y=path_unit_vector.x)

        # Transforms the start and end nodes based on the inverted unit vector
        # and the displacement magnitude

        transformed_start_node_xy = Point(
            x=start_node_xy.x + inv_path_unit_vector.x * abs(displacement),
            y=start_node_xy.y + inv_path_unit_vector.y * abs(displacement),
        )
        transformed_end_node_xy = Point(
            x=end_node_xy.x + inv_path_unit_vector.x * abs(displacement),
            y=end_node_xy.y + inv_path_unit_vector.y * abs(displacement),
        )

        return (transformed_start_node_xy, transformed_end_node_xy)

    def get_path_vector_and_yaw(
        self, image: np.ndarray, index: int
    ) -> tuple[tuple[Point, Point], float]:
        """
        Gets the path vector and estimated yaw from the image after manually selecting the edges of
        the path and given an estimated location.

        :param image: The image being passed through the system
        :param index: The index of the dataset that the system is currently on
        :return: The two points that forms a line which the system likely lies on
        """

        # Get the estimated position of the image and convert it to xy
        estimated_location_latlon = (
            self.dataset.rtk_lat.iloc[index],
            self.dataset.rtk_lon.iloc[index],
        )
        estimated_location_xy = Point(*self.converter.latlon_to_xy(estimated_location_latlon))
        estimated_yaw_deg = self.converter.heading_to_yaw(self.dataset.rtk_heading.iloc[index])

        # Make a copy of the image to mark on and initialize the points to be put on the image
        image_display = image.copy()
        self.points = []

        # Show the starting image to select the points
        print("Select Four Points corresponding to the edges of the path")
        print("""Follow the pattern of
                  - P1 = Bottom Left
                  - P2 = Top Left
                  - P3 = Bottom Right
                  - P4 = Top Right""")
        print("Enter 'Enter' to skip image")

        cv2.imshow(
            "Select Path Edges",
            image_display,
        )
        cv2.setMouseCallback("Select Path Edges", self.click_event, param=(image_display))

        # Loop until the edges of the path are marked and a key is entered
        while True:
            # Make a waitkey for when enter is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                # Gets the start and end point of the path based on the estimated location
                start_node_xy, end_node_xy, path_width, path_yaw = self.get_path_data(
                    estimated_location_latlon, estimated_location_xy, estimated_yaw_deg
                )

                # Check that 4 points were selected
                if len(self.points) == 4:
                    # Get the four points
                    p1, p2, p3, p4 = self.points

                    # Get the intersection between the 4 points which make 2 lines
                    intersection_point, m_left, m_right = self.get_intersection((p1, p2), (p3, p4))

                    # Gets the lateral displacement on the path and the yaw on the path
                    displacement, yaw = self.get_displacement_and_yaw(
                        intersection_point, m_left, m_right, path_width, path_yaw
                    )

                    # Gets the transformed vector that the system lies on
                    transformed_path_vector = self.get_path_vector(
                        start_node_xy, end_node_xy, displacement
                    )
                else:
                    yaw = estimated_yaw_deg
                    transformed_path_vector = (
                        start_node_xy,
                        end_node_xy,
                    )
                    displacement = "No Path Detected"
                    print("No path detected")

                # Destroys all the windows created by cv
                cv2.destroyAllWindows()

                return transformed_path_vector, yaw

            elif key == 27:
                cv2.destroyAllWindows()
                sys.exit()

                return None, None
