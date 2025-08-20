"""Tree Matcher script to match trees based on the ground position and the
angle of the trees seen from the ground.

file: tree_matcher.py
author: Cole Malinchock and Jack Elia
"""

import csv
import math

from converter import Converter
from data_structs import Point, Pose2d, Tree, Wedge

from project_sgil.constants import (
    AOI_ANGLE_DEG,
    AOI_RADIUS_M,
    HEADING_ERROR_DEG,
    ORIGIN,
    TREE_LOCATIONS_PATH,
)
from project_sgil.debug_visualizer import DebugVisualizer
from project_sgil.utils import distance, get_relative_angle, std_deviation_of_distances


class TreeMatcher:
    """Match trees based on satellite and ground view data to estimate vehicle
    position."""

    def __init__(self) -> None:
        """Initialize the TreeMatcher.

        :param: Initializes the converter and loads satellite tree
            locations.
        """

        self.satellite_tree_locations: list[Point] = []
        self.aoi_trees: list[Tree] = []
        self.wedges: list[Wedge] = []
        self.converter = Converter(ORIGIN[0], ORIGIN[1])

        # Load tree locations from CSV and convert to Point instances
        with open(TREE_LOCATIONS_PATH, newline="") as csvfile:
            scanner = csv.reader(csvfile, delimiter=",")
            for row in scanner:
                x, y = self.converter.latlon_to_xy((float(row[0]), float(row[1])))
                point = Point(x, y)
                self.satellite_tree_locations.append(point)

    @staticmethod
    def sequences_as_maps_from_wedges(
        wedges: list[Wedge],
        n: int,  # max length
        min_len: int = 1,  # minimum length to record
    ) -> list[dict[Wedge, Tree]]:
        """Enumerate skip-allowed, ordered wedge→tree selections as dicts.

        :param wedges: Ordered list of wedges to traverse.
        :param n: Maximum number of selections to include in a result.
        :param min_len: Minimum number of selections required for a
            result.
        :return: List of dictionaries mapping Wedge -> Tree (no repeated
            Tree ids).
        """
        results: list[dict[Wedge, Tree]] = []
        # Deduplicate identical results even if reached via different recursion paths.
        # Key is a frozenset of (wedge_index, tree_id) so order in the dict doesn't affect uniqueness.
        seen_maps: set[frozenset[tuple[int, int]]] = set()

        # Kick off the external backtracking routine.
        TreeMatcher._backtrack_wedge_maps(
            wedges=wedges,
            max_length=n,
            min_length=min_len,
            wedge_index=0,
            chosen_pairs=[],
            used_ids=set(),
            results=results,
            seen_maps=seen_maps,
        )
        return results

    @staticmethod
    def _backtrack_wedge_maps(
        wedges: list[Wedge],
        max_length: int,
        min_length: int,
        wedge_index: int,
        chosen_pairs: list[tuple[int, Tree]],
        used_ids: set[int],
        results: list[dict[Wedge, Tree]],
        seen_maps: set[frozenset[tuple[int, int]]],
    ) -> None:
        """Recursive generator for sequences_as_maps_from_wedges (defined
        externally).

        :param wedges: Ordered list of wedges to traverse.
        :param max_length: Hard cap on number of (wedge, tree)
            selections in one result.
        :param min_length: Minimum number of selections for a result to
            be recorded.
        :param wedge_index: Current index into the wedges list.
        :param chosen_pairs: Accumulated (wedge_index, Tree) selections
            so far.
        :param used_ids: Set of Tree ids already selected (prevents
            reuse across wedges).
        :param results: Output accumulator for dictionaries mapping
            Wedge -> Tree.
        :param seen_maps: Set used to deduplicate identical selections.
        """
        # Do not exceed the maximum selection length.
        if len(chosen_pairs) > max_length:
            return

        # Record any partial solution whose size is within [min_length, max_length].
        if min_length <= len(chosen_pairs) <= max_length:
            uniqueness_key = frozenset((idx, t.id) for idx, t in chosen_pairs)
            if uniqueness_key not in seen_maps:
                seen_maps.add(uniqueness_key)
                # Build the {Wedge -> Tree} dictionary from the stored indices.
                result_map: dict[Wedge, Tree] = {wedges[idx]: t for idx, t in chosen_pairs}
                results.append(result_map)
            # If we already reached the maximum allowed length, do not extend this path.
            if len(chosen_pairs) == max_length:
                # Note: we still return to explore sibling branches from earlier frames.
                pass

        # If we have traversed all wedges, there is nothing more to do.
        if wedge_index == len(wedges):
            return

        # Prune when even selecting from every remaining wedge cannot reach min_length.
        remaining_wedges = len(wedges) - wedge_index
        if len(chosen_pairs) + remaining_wedges < min_length:
            return

        current_wedge = wedges[wedge_index]

        # --- Option 1: pick exactly one eligible tree from this wedge ---
        # Build candidate list inline (combine trees + matched_tree, dedupe by tree id).
        candidate_trees: list[Tree] = []
        seen_ids_in_wedge: set[int] = set()

        # Trees listed directly in the wedge.
        for tree in current_wedge.trees:
            if tree.id not in seen_ids_in_wedge:
                seen_ids_in_wedge.add(tree.id)
                candidate_trees.append(tree)

        # Optional matched_tree, if present and not already included.
        if current_wedge.matched_tree is not None:
            mt = current_wedge.matched_tree
            if mt.id not in seen_ids_in_wedge:
                candidate_trees.append(mt)

        # Try selecting each candidate that does not reuse a previously chosen Tree id.
        for tree in candidate_trees:
            if tree.id in used_ids:
                continue
            used_ids.add(tree.id)
            chosen_pairs.append((wedge_index, tree))
            TreeMatcher._backtrack_wedge_maps(
                wedges=wedges,
                max_length=max_length,
                min_length=min_length,
                wedge_index=wedge_index + 1,
                chosen_pairs=chosen_pairs,
                used_ids=used_ids,
                results=results,
                seen_maps=seen_maps,
            )
            # Undo selection to explore alternative branches (classic backtracking).
            chosen_pairs.pop()
            used_ids.remove(tree.id)

        # --- Option 2: skip this wedge entirely (skip-allowed policy) ---
        TreeMatcher._backtrack_wedge_maps(
            wedges=wedges,
            max_length=max_length,
            min_length=min_length,
            wedge_index=wedge_index + 1,
            chosen_pairs=chosen_pairs,
            used_ids=used_ids,
            results=results,
            seen_maps=seen_maps,
        )

    def match_trees(self, current_pose: Pose2d, ground_thetas: list[float]) -> Point:
        """Match trees based on current position and ground view angles.

        :param current_pose: Current position and heading as Pose2d.
        :param ground_thetas: List of camera angles to trees in ground
            view (positive = left, negative = right).
        :return: Estimated location of the vehicle as Point.
        """

        # The area of interest based on the current pose
        self.aoi_trees = self.get_area_of_interest(current_pose)

        # Loop through all the thetas and make their corresponding wedges
        for theta in ground_thetas:
            self.wedges.append(self.create_wedge(current_pose, theta))

        # Estimate the location by matching the wedges to the identified trees
        estimated_location = self.wedge_matching(self.wedges, current_pose)

        # Return the estimated location
        return estimated_location

    def get_area_of_interest(self, current_pose: Pose2d) -> list[Tree]:
        """Identify satellite trees within area of interest.

        :param current_pose: Current position estimation as Pose2d.
        :return: List of tree locations (Point) within the area of
            interest.
        """

        area_of_interest_tree_locations: list[Tree] = []

        for tree_id, tree in enumerate(self.satellite_tree_locations):
            # Checks if the distance is within the radius
            if distance(tree, current_pose) < AOI_RADIUS_M:
                # Checks if the relative angle to the tree is within the expected limit
                rel_angle_deg = get_relative_angle(tree, current_pose)
                if abs(rel_angle_deg) < AOI_ANGLE_DEG:
                    area_of_interest_tree_locations.append(Tree(tree.x, tree.y, tree_id))

        return area_of_interest_tree_locations

    def create_wedge(self, current_pose: Pose2d, theta: float) -> Wedge:
        """Create a wedge based on current location and ground view angle.

        :param current_pose: Current position and orientation as Pose2d.
        :param theta: Ground view angle to a single tree.
        :return: Wedge object containing the theta and trees in the
            wedge.
        """

        wedge = Wedge(theta)

        for tree in self.aoi_trees:
            rel_angle_deg = get_relative_angle(tree, current_pose)
            if abs(rel_angle_deg - theta) < HEADING_ERROR_DEG:
                wedge.trees.append(tree)

        print(f"Wedge created with {len(wedge.trees)} trees")

        return wedge

    def wedge_matching(self, wedges: list[Wedge], current_pose: Pose2d) -> Point:
        """Estimate position by evaluating all skip-allowed wedge→tree maps
        (length ≥ 2) and picking the intersection point closest to
        current_pose."""

        # Generate maps: {Wedge -> Tree}, allowing skips, lengths in [2, len(wedges)]
        wedge_combinations = TreeMatcher.sequences_as_maps_from_wedges(
            wedges, n=len(wedges), min_len=2
        )

        best_pt: Point | None = None
        best_dist: float = float("inf")
        best_rms: float = float("inf")  # tie-breaker if distances are equal

        # Solve least-squares intersection for a set of lines defined by (point, angle)
        def ls_intersection(
            lines: list[tuple[Point, float]],
        ) -> tuple[float, float, float] | None:
            """
            Each line: (p_i, phi_i) where phi_i is direction angle in radians.
            Minimize sum_i (n_i·x - n_i·p_i)^2 with n_i = (-sin phi_i, cos phi_i).
            Returns (x, y, rms_residual) or None if ill-conditioned.
            """
            s11 = s12 = s22 = 0.0
            t1 = t2 = 0.0
            b_vals: list[float] = []
            normals: list[tuple[float, float]] = []

            for p_i, phi in lines:
                sn = math.sin(phi)
                cs = math.cos(phi)
                nx, ny = -sn, cs
                b_i = nx * p_i.x + ny * p_i.y

                s11 += nx * nx
                s12 += nx * ny
                s22 += ny * ny
                t1 += nx * b_i
                t2 += ny * b_i

                b_vals.append(b_i)
                normals.append((nx, ny))

            det = s11 * s22 - s12 * s12
            if abs(det) < 1e-12:
                return None  # lines are (near) parallel / degenerate

            inv11 = s22 / det
            inv12 = -s12 / det
            inv22 = s11 / det

            x = inv11 * t1 + inv12 * t2
            y = inv12 * t1 + inv22 * t2

            # RMS perpendicular residual to the set of lines
            sq_sum = 0.0
            for (nx, ny), b_i in zip(normals, b_vals, strict=False):
                r = nx * x + ny * y - b_i
                sq_sum += r * r
            m = len(normals)
            rms = math.sqrt(sq_sum / m) if m > 0 else float("inf")
            return (x, y, rms)

        base_yaw_deg = current_pose.yaw

        debug_visualizer = DebugVisualizer()

        i = 0

        for wedge_map in wedge_combinations:
            # Build the set of lines for this combination
            lines: list[tuple[Point, float]] = []
            for wedge, tree in wedge_map.items():
                phi = math.radians(base_yaw_deg + wedge.theta_degrees)
                lines.append((tree, phi))

            sol = ls_intersection(lines)
            if sol is None:
                continue

            x_hat, y_hat, rms = sol

            # Distance from the candidate intersection to the current pose
            dx = x_hat - current_pose.x
            dy = y_hat - current_pose.y
            dist = math.hypot(dx, dy)

            i += 1

            # Set all the wedge matched trees to the best point
            for wedge, tree in wedge_map.items():
                wedge.matched_tree = tree

            if len(wedge_map) == len(wedges):
                debug_visualizer.plot_wedges(
                    wedges,
                    current_pose,
                    self.aoi_trees,
                    Point(x_hat, y_hat),
                    f"combo_{i}_dist_{dist:.2f}",
                )

            # Choose the intersection closest to current_pose.
            # If distances tie (very rare), prefer the lower RMS fit as a tie-breaker.
            if (dist < best_dist) or (math.isclose(dist, best_dist) and rms < best_rms):
                best_dist = dist
                best_rms = rms
                best_pt = Point(x_hat, y_hat)

        # Fallback: if everything was degenerate, return the current pose
        if best_pt is None:
            return Point(current_pose.x, current_pose.y)
        return best_pt

    def create_vectors_from_wedges(
        self, wedges: list[Wedge], current_pose: Pose2d
    ) -> list[list[Point]]:
        """Create vectors from wedges for intersection analysis.

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
        """Find all intersection points between the vectors.

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
        """Find the intersection point of two vectors.

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
        """Calculate the mean centroid of a set of points.

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
        """Analyze intersections of vectors to find centroid and standard
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
