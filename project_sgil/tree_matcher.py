"""
Tree Matcher script to match trees based on the ground position and the angle of the trees seen from the ground

file: tree_matcher.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
import csv
import math
import statistics
from itertools import product
import matplotlib.pyplot as plt
from constants import *
from converter import Converter
from data_structs import *
from shapely.geometry import LineString, Point

# Import custom data structs
from data_structs import Wedge, Pose2d


class TreeMatcher:
    """
    Match trees based on satellite and ground view data to estimate vehicle
    position.
    """

    def __init__(self) -> None:
        """
        Initialize the TreeMatcher.

        :param: Initializes the converter and loads satellite tree
            locations.
        """

        self.all_sat_tree_loc = []
        self.converter = Converter(ORIGIN[0], ORIGIN[1])

        # Load tree locations from CSV and convert to (x, y) coordinates
        with open(TREE_LOCATIONS_PATH, newline="") as csvfile:
            scanner = csv.reader(csvfile, delimiter=",")
            for row in scanner:
                point = self.converter.latlon_to_xy((float(row[0]), float(row[1])))
                self.all_sat_tree_loc.append(point)


    def match_trees(self, current_pose: Pose2d, ground_thetas: float) -> tuple[float, float]:
        """
        Match trees based on current position and ground view angles.

        :param current_pose: Current position and heading as (x, y,
            heading).
        :param ground_thetas: List of camera angles to trees in ground
            view (negative = left, positive = right).
        :return: Estimated location of the vehicle.
        """

        # The area of interest based on the current pose
        aoi_sat_trees = self.get_area_of_interest(current_pose)

        # Loop through all the thetas and make their corresponding wedges
        wedges = []
        for theta in ground_thetas:
            wedges.append(self.create_wedge(current_pose, aoi_sat_trees, theta))

        # Estimate the location by matching the wedges to the identified trees
        estimated_location = self.wedge_matching(wedges, current_pose)

        # Return the estimated location
        return estimated_location


    def get_area_of_interest(self, current_pose: Pose2d) -> list[tuple[float, float]]:
        """
        Identify satellite trees within area of interest.

        :param current_pose: Current position estimation as (x, y,
            heading).
        :return: List of tree locations within the area of interest.
        """

        # Initializes the list of trees in the area of interest
        area_of_interest_tree_loc = []

        # Loops through all the trees in the list to find the trees that are within the area of interest
        for tree in self.all_sat_tree_loc:

            # Checks if the distance is within the radius
            if self.distance(tree, (current_pose.x, current_pose.y)) < AOI_RADIUS_M:

                # Checks if the relative angle to the tree is within the expected limit
                rel_angle_deg = self.get_relative_angle(tree, current_pose)
                if abs(rel_angle_deg) < AOI_ANGLE_DEG:
                    area_of_interest_tree_loc.append(tree)

        # Check if there is a plot to show
        if PLOT:
            self.debug_plot_aoi(self.all_sat_tree_loc, area_of_interest_tree_loc, current_pose)

        return area_of_interest_tree_loc


    def debug_plot_aoi(self, all_sat_tree_loc: list[tuple[float, float]], aoi_sat_trees: list[tuple[float, float]], current_pose: Pose2d) -> None:
        """
        Visualize AOI trees for debugging.

        :param all_sat_tree_loc: List of all satellite tree locations.
        :param aoi_sat_trees: List of trees within the area of interest.
        :param current_pose: Current position (x, y, heading).
        """

        # Creates the figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all satellite trees in blue
        all_x = [tree[0] for tree in all_sat_tree_loc]
        all_y = [tree[1] for tree in all_sat_tree_loc]
        ax.scatter(all_x, all_y, s=20, c="blue", alpha=0.5, label="All Trees")

        # Plot AOI trees in green
        if aoi_sat_trees and len(aoi_sat_trees) > 0:
            aoi_x = [tree[0] for tree in aoi_sat_trees]
            aoi_y = [tree[1] for tree in aoi_sat_trees]
            ax.scatter(aoi_x, aoi_y, s=50, c="green", alpha=0.8, label="AOI Trees")

        # Plot current position in red
        if current_pose:
            ax.scatter(
                current_pose.x,
                current_pose.y,
                s=100,
                c="red",
                marker="*",
                label="Current Position",
            )

            # Draw line showing heading direction
            heading_rad = math.radians(current_pose.yaw)
            arrow_length = 5
            dx = arrow_length * math.cos(heading_rad)
            dy = arrow_length * math.sin(heading_rad)
            ax.arrow(
                current_pose.x,
                current_pose.y,
                dx,
                dy,
                head_width=1,
                head_length=2,
                fc="red",
                ec="red",
            )

        # Add AOI radius circle
        if current_pose:
            circle = plt.Circle(
                (current_pose.x, current_pose.y),
                AOI_RADIUS_M,
                fill=False,
                color="red",
                linestyle="--",
                alpha=0.7,
            )
            ax.add_patch(circle)

        # Add field of view indicator
        if current_pose:
            heading_rad = math.radians(current_pose.yaw)
            half_fov = math.radians(AOI_ANGLE_DEG)
            start_angle = heading_rad - half_fov
            end_angle = heading_rad + half_fov

            wedge = plt.matplotlib.patches.Wedge(
                (current_pose.x, current_pose.y),
                AOI_RADIUS_M,
                math.degrees(start_angle),
                math.degrees(end_angle),
                fill=False,
                color="green",
                linestyle=":",
                alpha=0.7,
            )
            ax.add_patch(wedge)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add labels and legend
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Area of Interest Debug Plot")
        ax.legend()

        # Add stats
        stats_text = (
            f"Total trees: {len(all_sat_tree_loc)}\n"
            f"AOI trees: {len(aoi_sat_trees) if aoi_sat_trees else 0}\n"
            f"Position: ({current_pose.x:.1f}, {current_pose.y:.1f})\n"
            f"Heading: {current_pose.yaw:.1f}°"
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def create_wedge(self, current_pose: Pose2d, satellite_trees: list[tuple[float, float]], theta: float) -> Wedge:
        """
        Create a wedge based on current location and ground view angle.

        :param current_pose: Current position and orientation as (x, y,
            heading).
        :param satellite_trees: List of trees in the AOI.
        :param theta: Ground view angle to a single tree.
        :return: Wedge object containing the theta and trees in the
            wedge.
        """

        # Create a wedge based on the given theta
        wedge = Wedge(theta)

        # Loop through all the trees in the satellite trees to see if they lie within the wedge
        for tree in satellite_trees:
            rel_angle_deg = self.get_relative_angle(tree, current_pose)
            if abs(rel_angle_deg - theta) < HEADING_ERROR_DEG:

                wedge.trees.append(Tree(tree[0], tree[1], -1))

        return wedge


    def wedge_matching(self, wedges: list[Wedge], current_pose: Pose2d) -> tuple[float, float]:
        """
        Match wedges to trees to find the most accurate position.

        :param wedges: List of wedges containing available trees.
        :param current_pose: Current position estimate as (x, y,
            heading).
        :return: Estimated position of the vehicle as (x, y).
        """

        # Two lists for holding the matched and unmatched combinations of the wedges
        matched_wedges = []
        unmatched_wedges = []

        for wedge in wedges:
            if len(wedge.trees) == 1:
                matched_wedges.append(wedge)
            elif len(wedge.trees) > 1:
                unmatched_wedges.append(wedge)

        if len(unmatched_wedges) == 0 and len(matched_wedges) >= 2:
            vectors = self.create_vectors_from_wedges(matched_wedges, current_pose)
            centroid, _, std_dev = self.analyze_vector_intersections(vectors)
            return centroid

        wedge_combinations = [wedge.trees for wedge in unmatched_wedges]

        if not wedge_combinations:
            return (current_pose.x, current_pose.y)

        all_wedge_combinations = list(product(*wedge_combinations))
        unique_wedge_combinations = [
            combo for combo in all_wedge_combinations if len(set(tree.id for tree in combo)) == len(combo)
        ]

        # If there is not a unique wedge combo, return the current position
        if not unique_wedge_combinations:
            return (current_pose.x, current_pose.y)

        # Initializes the lowest standard deviation
        lowest_std_dev = float("inf")
        best_centroid = None

        for tree_combo in unique_wedge_combinations:

            # Create vectors from the current position to each tree in the combo
            vectors = []

            for i, _tree in enumerate(tree_combo):
                if i < len(unmatched_wedges):
                    wedge = unmatched_wedges[i]
                    start_point = (current_pose.x, current_pose.y)
                    heading_rad = math.radians(current_pose.yaw)
                    theta_rad = math.radians(wedge.theta_deg)
                    direction_rad = heading_rad - theta_rad
                    vector_length = 100
                    end_x = current_pose.x + vector_length * math.cos(direction_rad)
                    end_y = current_pose.y + vector_length * math.sin(direction_rad)
                    vectors.append([start_point, (end_x, end_y)])

            for wedge in matched_wedges:
                if wedge.trees:
                    start_point = (current_pose.x, current_pose.y)
                    theta_rad = math.radians(wedge.theta_deg)
                    heading_rad = math.radians(current_pose.yaw)
                    direction_rad = heading_rad - theta_rad
                    vector_length = 100
                    end_x = current_pose.x + vector_length * math.cos(direction_rad)
                    end_y = current_pose.y + vector_length * math.sin(direction_rad)
                    vectors.append([start_point, (end_x, end_y)])

            if len(vectors) >= 2:
                centroid, intersections, std_dev = self.analyze_vector_intersections(vectors)
                if centroid and std_dev is not None and std_dev < lowest_std_dev:
                    lowest_std_dev = std_dev
                    best_centroid = centroid

        # If there is a best centroid return that
        # Else return the current position
        if best_centroid:
            return best_centroid
        return (current_pose.x, current_pose.y)


    def create_vectors_from_wedges(self, wedges: list[Wedge], current_pose: Pose2d) -> list[tuple[float, float]]:
        """
        Create vectors from wedges for intersection analysis.

        :param wedges: List of wedges with trees.
        :param current_pose: Current position estimate as (x, y, heading).
        :return: List of vectors as [[start_point, end_point], ...].
        """

        # Initializes the list of vectors
        vectors = []

        # Loops through all the wedges
        for wedge in wedges:
            if wedge.trees:
                start_point = (current_pose.x, current_pose.y)
                theta_rad = math.radians(wedge.theta_deg)
                heading_rad = math.radians(current_pose.yaw)
                direction_rad = heading_rad - theta_rad

                vector_length = 100
                end_x = current_pose.x + vector_length * math.cos(direction_rad)
                end_y = current_pose.y + vector_length * math.sin(direction_rad)
                vectors.append([start_point, (end_x, end_y)])
                
        return vectors

    def find_intersections(self, vectors: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Find all intersection points between the vectors.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :return: List of intersection points as (x, y) tuples.
        """

        # Gets the list of lines from the vectors and initializes the list of intersection
        lines = [LineString(vec) for vec in vectors]
        intersections = []

        # Loops through each line for an intersection
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if i < j and line1.intersects(line2):
                    inter = line1.intersection(line2)
                    if isinstance(inter, Point):
                        intersections.append((inter.x, inter.y))

        return intersections


    def mean_centroid(self, points: list[tuple[float, float]]) -> tuple[float, float]:
        """
        Calculate the mean centroid of a set of points.

        :param points: List of points as (x, y) tuples.
        :return: Centroid as (x, y) tuple or None if empty list.
        """

        # Checks if there are any points
        if not points:
            return None
        
        # Gets all the x and y coordinates from each point
        x_coords = [pt[0] for pt in points]
        y_coords = [pt[1] for pt in points]

        # Calculates the mean centroid of all the points
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))


    def std_deviation_of_distances(self, points: list[tuple[float, float]], centroid: tuple[float, float]) -> float:
        """
        Calculate standard deviation of distances from points to centroid.

        :param points: List of points as (x, y) tuples.
        :param centroid: Centroid point as (x, y) tuple.
        :return: Standard deviation of distances or None if insufficient
            data.
        """

        # Checks that there is a list of points and there is a centroid
        if not points or centroid is None:
            return None
        
        # Gets the distance of each point to the centroid
        distances = [self.distance(pt, centroid) for pt in points]

        # Calculates the standard deviation of all the distances
        return statistics.stdev(distances) if len(distances) > 1 else 0.0


    def analyze_vector_intersections(self, vectors: list[tuple[float, float]]) -> tuple[tuple[float, float], list[tuple[float, float]], float]:
        """
        Analyze intersections of vectors to find centroid and standard
        deviation.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :return: Tuple of (centroid, intersections, standard_deviation).
        """

        # Gets the intersections from all the vectors 
        intersections = self.find_intersections(vectors)

        # Checks if there are any intersections
        if not intersections:
            print("No intersections found.")
            return None, None, None

        # Gets the centroid and standard deviation
        centroid = self.mean_centroid(intersections)
        std_dev = self.std_deviation_of_distances(intersections, centroid)

        return centroid, intersections, std_dev


    def distance(self, point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.

        :param point1: First point as (x, y) tuple.
        :param point2: Second point as (x, y) tuple.
        :return: Distance between the points.
        """

        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])
      

    def get_relative_angle(self, point: tuple[float, float], current_pose: Pose2d) -> float:
        """
        Calculate relative angle from current heading to a point.

        :param point: Target point as (x, y) tuple.
        :param current_pose: Current position as (x, y, heading) tuple.
        :return: Relative angle in degrees (-180, 180).
        """

        current_x, current_y, current_heading = current_pose.x, current_pose.y, current_pose.yaw
        target_x, target_y = point
        abs_angle_rad = math.atan2(target_y - current_y, target_x - current_x)
        abs_angle_deg = math.degrees(abs_angle_rad)
        rel_angle_deg = abs_angle_deg - current_heading

        rel_angle_deg = ((rel_angle_deg + 180) % 360) - 180
        return rel_angle_deg
