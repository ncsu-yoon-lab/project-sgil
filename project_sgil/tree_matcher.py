"""
Tree Matcher script to match trees based on the ground position and the
angle of the trees seen from the ground.

file: tree_matcher.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
import csv
import math
from itertools import product

from constants import *
from converter import Converter
from data_structs import Point, Pose2d, Tree, Wedge

from project_sgil.utils import distance, get_relative_angle, std_deviation_of_distances


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

        self.all_sat_tree_loc: list[Point] = []
        self.aoi_sat_trees: list[Point] = []
        self.wedges: list[Wedge] = []
        self.converter = Converter(ORIGIN[0], ORIGIN[1])

        # Load tree locations from CSV and convert to Point instances
        with open(TREE_LOCATIONS_PATH, newline="") as csvfile:
            scanner = csv.reader(csvfile, delimiter=",")
            for row in scanner:
                x, y = self.converter.latlon_to_xy((float(row[0]), float(row[1])))
                point = Point(x, y)
                self.all_sat_tree_loc.append(point)

    def match_trees(self, current_pose: Pose2d, ground_thetas: list[float]) -> Point:
        """
        Match trees based on current position and ground view angles.

        :param current_pose: Current position and heading as Pose2d.
        :param ground_thetas: List of camera angles to trees in ground
            view (positive = left, negative = right).
        :return: Estimated location of the vehicle as Point.
        """

        # The area of interest based on the current pose
        self.aoi_sat_trees = self.get_area_of_interest(current_pose)

        # Loop through all the thetas and make their corresponding wedges
        for theta in ground_thetas:
            self.wedges.append(self.create_wedge(current_pose, self.aoi_sat_trees, theta))

        # Estimate the location by matching the wedges to the identified trees
        estimated_location = self.wedge_matching(self.wedges, current_pose)

        # Return the estimated location
        return estimated_location

    def get_area_of_interest(self, current_pose: Pose2d) -> list[Point]:
        """
        Identify satellite trees within area of interest.

        :param current_pose: Current position estimation as Pose2d.
        :return: List of tree locations (Point) within the area of
            interest.
        """

        area_of_interest_tree_loc: list[Point] = []

        for tree in self.all_sat_tree_loc:
            # Checks if the distance is within the radius
            if distance(tree, current_pose) < AOI_RADIUS_M:
                # Checks if the relative angle to the tree is within the expected limit
                rel_angle_deg = get_relative_angle(tree, current_pose)
                if abs(rel_angle_deg) < AOI_ANGLE_DEG:
                    area_of_interest_tree_loc.append(tree)

        return area_of_interest_tree_loc

    def create_wedge(
        self, current_pose: Pose2d, satellite_trees: list[Point], theta: float
    ) -> Wedge:
        """
        Create a wedge based on current location and ground view angle.

        :param current_pose: Current position and orientation as Pose2d.
        :param satellite_trees: List of trees in the AOI as Point
            instances.
        :param theta: Ground view angle to a single tree.
        :return: Wedge object containing the theta and trees in the
            wedge.
        """

        wedge = Wedge(theta)
        tree_idx = 0

        for tree in satellite_trees:
            rel_angle_deg = get_relative_angle(tree, current_pose)
            if abs(rel_angle_deg - theta) < HEADING_ERROR_DEG:
                wedge.trees.append(Tree(tree.x, tree.y, tree_idx))
                tree_idx += 1

        print(f"Wedge created with {len(wedge.trees)} trees")

        return wedge

    def wedge_matching(self, wedges: list[Wedge], current_pose: Pose2d) -> Point:
        """
        Match wedges to trees to find the most accurate position.

        :param wedges: List of wedges containing available trees.
        :param current_pose: Current position estimate as Pose2d.
        :return: Estimated position of the vehicle as Point.
        """

        matched_wedges: list[Wedge] = []
        unmatched_wedges: list[Wedge] = []

        for wedge in wedges:
            if len(wedge.trees) == 1:
                wedge.matched_tree = wedge.trees[0]
                matched_wedges.append(wedge)
            elif len(wedge.trees) > 1:
                unmatched_wedges.append(wedge)

        if not unmatched_wedges and len(matched_wedges) >= 2:
            vectors = self.create_vectors_from_wedges(matched_wedges, current_pose)
            centroid, _, std_dev = self.analyze_vector_intersections(vectors)
            return centroid

        wedge_combinations = [wedge.trees for wedge in unmatched_wedges]
        if not wedge_combinations:
            print("No wedge found")
            return Point(current_pose.x, current_pose.y)

        all_wedge_combinations = list(product(*wedge_combinations))
        unique_wedge_combinations = [
            combo
            for combo in all_wedge_combinations
            if len({tree.id for tree in combo}) == len(combo)
        ]

        print(unique_wedge_combinations)

        lowest_std_dev = float("inf")
        best_centroid: Point | None = None
        best_combo = None

        for tree_combo in unique_wedge_combinations:
            vectors: list[list[Point]] = []
            for i, tree in enumerate(tree_combo):
                if i < len(unmatched_wedges):
                    wedge = unmatched_wedges[i]
                    heading_rad = math.radians(current_pose.yaw)
                    theta_rad = math.radians(-wedge.theta_degrees)
                    direction_rad = (heading_rad - theta_rad - math.radians(180)) % (2 * math.pi)
                    vector_length = 100
                    start = Point(tree.x, tree.y)
                    end = Point(
                        tree.x + vector_length * math.cos(direction_rad),
                        tree.y + vector_length * math.sin(direction_rad),
                    )
                    vectors.append([start, end])

            for wedge in matched_wedges:
                if wedge.matched_tree:
                    tree = wedge.matched_tree
                    heading_rad = math.radians(current_pose.yaw)
                    theta_rad = math.radians(-wedge.theta_degrees)
                    direction_rad = (heading_rad - theta_rad - math.radians(180)) % (2 * math.pi)
                    vector_length = 100
                    start = Point(tree.x, tree.y)
                    end = Point(
                        tree.x + vector_length * math.cos(direction_rad),
                        tree.y + vector_length * math.sin(direction_rad),
                    )
                    vectors.append([start, end])

            if len(vectors) >= 2:
                centroid, intersections, std_dev = self.analyze_vector_intersections(vectors)
                if centroid and std_dev is not None and std_dev < lowest_std_dev:
                    best_combo = tree_combo
                    lowest_std_dev = std_dev
                    best_centroid = centroid

        if best_centroid and best_combo:
            combo_idx = 0
            for wedge in wedges:
                if wedge.matched_tree is None:
                    wedge.matched_tree = best_combo[combo_idx]
                    combo_idx += 1
            self.wedges = wedges
            return best_centroid

        print("No best centroid")
        return Point(current_pose.x, current_pose.y)

    def create_vectors_from_wedges(
        self, wedges: list[Wedge], current_pose: Pose2d
    ) -> list[list[Point]]:
        """
        Create vectors from wedges for intersection analysis.

        :param wedges: List of wedges with trees.
        :param current_pose: Current position estimate as Pose2d.
        :return: List of vectors as [[start_point, end_point], ...].
        """

        vectors: list[list[Point]] = []
        for wedge in wedges:
            if wedge.trees:
                theta_rad = math.radians(wedge.theta_degrees)
                heading_rad = math.radians(current_pose.yaw)
                direction_rad = heading_rad - theta_rad

                vector_length = 100
                start = Point(current_pose.x, current_pose.y)
                end = Point(
                    current_pose.x + vector_length * math.cos(direction_rad),
                    current_pose.y + vector_length * math.sin(direction_rad),
                )
                vectors.append([start, end])

        return vectors

    def find_intersections(self, vectors: list[list[Point]]) -> list[Point]:
        """
        Find all intersection points between the vectors.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :return: List of intersection points as Point.
        """

        intersections: list[Point] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                intersection = self.find_intersection(vectors[i], vectors[j])
                if intersection:
                    intersections.append(intersection)
        return intersections

    def find_intersection(self, vector1: list[Point], vector2: list[Point]) -> Point | None:
        """
        Find the intersection point of two vectors.

        :param vector1: First vector as [start_point, end_point].
        :param vector2: Second vector as [start_point, end_point].
        :return: Intersection point as Point or None if no intersection.
        """

        p1, p2 = vector1
        p3, p4 = vector2
        delta1_x = p2.x - p1.x
        delta1_y = p2.y - p1.y
        delta2_x = p4.x - p3.x
        delta2_y = p4.y - p3.y

        m1 = delta1_y / delta1_x if delta1_x != 0 else float("inf")
        m2 = delta2_y / delta2_x if delta2_x != 0 else float("inf")
        b1 = p1.y - m1 * p1.x
        b2 = p3.y - m2 * p3.x

        if m1 == m2:
            return None

        x_int = (b2 - b1) / (m1 - m2)
        y_int = m1 * x_int + b1
        return Point(x_int, y_int)

    def mean_centroid(self, points: list[Point]) -> Point | None:
        """
        Calculate the mean centroid of a set of points.

        :param points: List of Point.
        :return: Centroid as Point or None if empty list.
        """

        if not points:
            return None
        x_coords = [pt.x for pt in points]
        y_coords = [pt.y for pt in points]
        return Point(sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def analyze_vector_intersections(
        self, vectors: list[list[Point]]
    ) -> tuple[Point | None, list[Point] | None, float | None]:
        """
        Analyze intersections of vectors to find centroid and standard
        deviation.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :return: Tuple of (centroid, intersections, standard_deviation).
        """

        intersections = self.find_intersections(vectors)
        if not intersections:
            print("No intersections found.")
            return None, None, None

        centroid = self.mean_centroid(intersections)
        std_dev = std_deviation_of_distances(intersections, centroid) if centroid else None
        return centroid, intersections, std_dev
