import csv
import math
import statistics
from itertools import product

import matplotlib.pyplot as plt
from constants import *
from converter import Converter
from data_structs import *
from shapely.geometry import LineString, Point

from project_sgil.utils import distance, get_relative_angle


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
        self.aoi_sat_trees = []
        self.wedges = []
        self.converter = Converter(ORIGIN[0], ORIGIN[1])

        # Load tree locations from CSV and convert to (x, y) coordinates
        with open(TREE_LOCATIONS_PATH, newline="") as csvfile:
            scanner = csv.reader(csvfile, delimiter=",")
            for idx, row in enumerate(scanner):
                # unpack lat/lon → x,y
                x, y = self.converter.latlon_to_xy((float(row[0]), float(row[1])))
                # now store a Tree (has .x and .y)
                self.tree_satellite_locations.append(Tree(x=x, y=y, id=idx))

    def match_trees(self, current_pose: Pose2d, ground_thetas: list[float]) -> Point:
        """
        Match trees based on current position and ground view angles.

        :param current_pose: Preliminary position estimation as a Pose2d
        :param ground_thetas: List of camera angles to trees in ground
            view (positive = left, negative = right).
        :return: Estimated location of the vehicle.
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

        :param current_pose: Current position estimation as (x, y,
            heading).
        :return: List of tree locations within the area of interest.
        """
        area_of_interest_tree_loc = []

        for tree in self.tree_satellite_locations:
            if distance(tree, current_pose) < AOI_RADIUS_M:
                rel_angle_deg = get_relative_angle(tree, current_pose)
                if abs(rel_angle_deg) < AOI_ANGLE_DEG:
                    area_of_interest_tree_loc.append(tree)

        return area_of_interest_tree_loc


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

        for tree in satellite_trees:
            rel_angle_deg = self.get_relative_angle(tree, current_pose)
            
            if abs(rel_angle_deg - theta) < HEADING_ERROR_DEG:
                wedge.trees.append(Tree(tree[0], tree[1], -1))

        print(f"Wedge created with {len(wedge.trees)} trees")

        return wedge

    def wedge_matching(self, wedges, current_pose):
        """
        Match wedges to trees to find the most accurate position.

        :param wedges: List of wedges containing available trees.
        :param current_pose: Current position estimate as (x, y,
            heading).
        :return: Estimated position of the vehicle as (x, y).
        """
        matched_wedges = []
        unmatched_wedges = []

        # Loops through all the wedges and separates them into matched and unmatched
        for wedge in wedges:
            if len(wedge.trees) == 1:
                matched_wedges.append(wedge)
                wedge.matched_tree = wedge.trees[0]
            elif len(wedge.trees) > 1:
                unmatched_wedges.append(wedge)
                
        # If there are no unmatched wedges, but at least two matched wedges, create vectors and analyze intersections
        if len(unmatched_wedges) == 0 and len(matched_wedges) >= 2:
            vectors = self.create_vectors_from_wedges(matched_wedges, current_pose)
            centroid, _, std_dev = self.analyze_vector_intersections(vectors)

            return centroid

        # Loops through all the unmatched wedges and creates a list of combinations of trees
        wedge_combinations = [wedge.trees for wedge in unmatched_wedges]
        
        # if there are no wedge combinations found, return the current position
        if not wedge_combinations:
            print("No wedge found")
            return (current_pose.x, current_pose.y)

        # Create all the combinations of wedges and filter for unique combinations
        all_wedge_combinations = list(product(*wedge_combinations))
        unique_wedge_combinations = [
            combo for combo in all_wedge_combinations if len(set(combo)) == len(combo)
        ]

        if not unique_wedge_combinations:
            print("No unique wedge found")
            return (current_pose.x, current_pose.y)

        lowest_std_dev = float("inf")
        best_centroid = None
        best_combo = None
        
        # Loops through each unique wedge combination
        for tree_combo in unique_wedge_combinations:
            vectors = []

            # Create vectors for each tree in the combo and matched wedges
            for i, tree in enumerate(tree_combo):
                
                if i < len(unmatched_wedges):

                    wedge = unmatched_wedges[i]
                    start_point = (tree.x, tree.y)
                    heading_rad = math.radians(current_pose.yaw)
                    theta_rad = math.radians(wedge.theta_deg * -1.0)
                    direction_rad = (heading_rad - theta_rad - math.radians(180)) % 360
                    vector_length = 100
                    end_x = tree.x + vector_length * math.cos(direction_rad)
                    end_y = tree.y + vector_length * math.sin(direction_rad)
                    # print("Unmatched heading: ", math.degrees(heading_rad))
                    # print("Unmatched theta: ", math.degrees(theta_rad))
                    # print("Unmatched altered heading: ", math.degrees(direction_rad))
                    # print("Unmatched start x: ", start_point[0])
                    # print("Unmatched start y: ", start_point[1])
                    vectors.append([start_point, (end_x, end_y)])

            for wedge in matched_wedges:
                if wedge.matched_tree:
                    
                    start_point = (wedge.matched_tree.x, wedge.matched_tree.y)
                    theta_rad = math.radians(wedge.theta_deg * -1.0)
                    heading_rad = math.radians(current_pose.yaw)
                    direction_rad = (heading_rad - theta_rad - math.radians(180)) % 360
                    vector_length = 100
                    end_x = wedge.matched_tree.x + vector_length * math.cos(direction_rad)
                    end_y = wedge.matched_tree.y + vector_length * math.sin(direction_rad)
                    vectors.append([start_point, (end_x, end_y)])
                    
            if len(vectors) >= 2:
                
                centroid, intersections, std_dev = self.analyze_vector_intersections(vectors)
                if centroid and std_dev is not None and std_dev < lowest_std_dev:
                    best_combo = tree_combo
                    lowest_std_dev = std_dev
                    best_centroid = centroid
                    
        # If there is a best centroid return that
        # Else return the current position
        if best_centroid and best_combo:

            # Match the combinations with their wedges
            combo_idx = 0
            for i in range(len(wedges)):
                if wedges[i].matched_tree is None:
                    wedges[i].matched_tree = best_combo[combo_idx]
                    combo_idx += 1
            
            self.wedges = wedges

            return best_centroid
        
        print("No best centroid")
        return (current_pose.x, current_pose.y)

    def create_vectors_from_wedges(self, wedges, current_pose):
        """
        Create vectors from wedges for intersection analysis.

        :param wedges: List of wedges with trees.
        :param current_pose: Current position estimate as (x, y, heading).
        :return: List of vectors as [[start_point, end_point], ...].
        """
        vectors = []
        for wedge in wedges:
            if wedge.trees:
                start_point = (current_pose[0], current_pose[1])
                theta_rad = math.radians(wedge.ground_theta_deg)
                heading_rad = math.radians(current_pose[2])
                direction_rad = heading_rad - theta_rad
                vector_length = 100
                end_x = current_pose[0] + vector_length * math.cos(direction_rad)
                end_y = current_pose[1] + vector_length * math.sin(direction_rad)
                vectors.append([start_point, (end_x, end_y)])
        return vectors

    def find_intersections(self, vectors):
        """
        Find all intersection points between the vectors.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :return: List of intersection points as (x, y) tuples.
        """
        intersections = []
        
        for i in range(len(vectors)):
            j = i
            while(j < len(vectors)):
                
                if i != j:
                    # print(f"Combo: ({vectors[i]}, {vectors[j]}")
                    intersections.append(self.find_intersection(vectors[i], vectors[j]))
                j += 1
                        
        return intersections
    
    def find_intersection(self, vector1, vector2):
        delta1_x = vector1[1][0] - vector1[0][0]
        delta1_y = vector1[1][1] - vector1[0][1]

        delta2_x = vector2[1][0] - vector2[0][0]
        delta2_y = vector2[1][1] - vector2[0][1]

        if delta1_x == 0:
            m1 = 9999.9
        else:
            m1 = delta1_y / delta1_x
        
        if delta2_x == 0:
            m2 = 9999.9
        else:
            m2 = delta2_y / delta2_x
        
        b1 = - (vector1[0][0] * m1) + vector1[0][1]
        b2 = - (vector2[0][0] * m2) + vector2[0][1]

        x_int = (b1 - b2) / (m2 - m1)
        y_int = m1 * x_int + b1

        intersection = (x_int, y_int)

        return intersection

    def mean_centroid(self, points):
        """
        Calculate the mean centroid of a set of points.

        :param points: List of points as (x, y) tuples.
        :return: Centroid as (x, y) tuple or None if empty list.
        """
        if not points:
            return None
        x_coords = [pt[0] for pt in points]
        y_coords = [pt[1] for pt in points]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def std_deviation_of_distances(self, points, centroid):
        """
        Calculate standard deviation of distances from points to centroid.

        :param points: List of points as (x, y) tuples.
        :param centroid: Centroid point as (x, y) tuple.
        :return: Standard deviation of distances or None if insufficient
            data.
        """
        if not points or centroid is None:
            return None
        distances = [self.distance(pt, centroid) for pt in points]
        return statistics.stdev(distances) if len(distances) > 1 else 0.0

    def analyze_vector_intersections(self, vectors):
        """
        Analyze intersections of vectors to find centroid and standard
        deviation.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :return: Tuple of (centroid, intersections, standard_deviation).
        """
        intersections = self.find_intersections(vectors)
        
        # Checks if there are any intersections
        if not intersections:
            print("No intersections found.")
            return None, None, None

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
