"""Tree Matcher script to match trees based on the ground position and the
angle of the trees seen from the ground.

file: tree_matcher.py
author: Cole Malinchock and Jack Elia
"""

import csv
import math

from converter import Converter
from data_structs import Point, Pose2d, Tree, Wedge, PoseEstimate

from project_sgil.constants import (
    AOI_ANGLE_DEG,
    AOI_RADIUS_M,
    HEADING_ERROR_DEG,
    ORIGIN,
    TREE_LOCATIONS_PATH,
)
from project_sgil.debug_visualizer import DebugVisualizer
from project_sgil.utils import distance, get_relative_angle


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
    def _generate_wedge_combinations(
        wedges: list[Wedge],
        n: int,  # max length
        min_len: int = 1,  # minimum length to record
    ) -> list[dict[Wedge, Tree]]:
        """Gets a dictionary mapping Wedge -> Tree for all possible
        combinations of selecting up to n trees from the wedges, allowing
        skipping wedges, and ensuring no Tree is used more than once per
        combination.

        :param wedges: Ordered list of wedges to traverse.
        :param n: Maximum number of selections to include in a result.
        :param min_len: Minimum number of selections required for a
            result.
        :return: List of dictionaries mapping Wedge -> Tree (no repeated
            Tree ids).
        """
        results: list[dict[Wedge, Tree]] = []
        # Deduplicate identical results even if reached via different recursion paths.
        # Key is a frozenset of (wedge_index, tree_id) so the order doesn't affect uniqueness.
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

    @staticmethod
    def _solve_least_squares_intersection(
        lines: list[tuple[Point, float]],
    ) -> tuple[float, float, float] | None:
        """Solve least-squares intersection of lines.

        :param lines: Each line defined by (point, angle_in_radians).
        :return: (x, y, rms_residual) of intersection, or None if lines are degenerate.
        """
        s11 = s12 = s22 = 0.0
        t1 = t2 = 0.0
        b_vals: list[float] = []
        normals: list[tuple[float, float]] = []

        for point, phi in lines:
            sn = math.sin(phi)
            cs = math.cos(phi)
            nx, ny = -sn, cs
            b_i = nx * point.x + ny * point.y

            s11 += nx * nx
            s12 += nx * ny
            s22 += ny * ny
            t1 += nx * b_i
            t2 += ny * b_i

            b_vals.append(b_i)
            normals.append((nx, ny))

        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-12:
            return None  # lines are nearly parallel / degenerate

        inv11 = s22 / det
        inv12 = -s12 / det
        inv22 = s11 / det

        x = inv11 * t1 + inv12 * t2
        y = inv12 * t1 + inv22 * t2

        # Compute RMS residual error
        sq_sum = 0.0
        for (nx, ny), b_i in zip(normals, b_vals, strict=False):
            r = nx * x + ny * y - b_i
            sq_sum += r * r
        m = len(normals)
        rms = math.sqrt(sq_sum / m) if m > 0 else float("inf")

        return x, y, rms

    def match_trees(self, current_pose: Pose2d, ground_thetas: list[float]) -> Point:
        """Match trees based on current position and ground view angles.

        :param current_pose: Current position and heading as Pose2d.
        :param ground_thetas: List of camera angles to trees in ground
            view (positive = left, negative = right).
        :return: Estimated location of the vehicle as Point.
        """

        # The area of interest based on the current pose
        self.aoi_trees = self._get_area_of_interest(current_pose)

        # Loop through all the thetas and make their corresponding wedges
        for theta in ground_thetas:
            self.wedges.append(self._create_wedge(current_pose, theta))

        # Estimate the location by matching the wedges to the identified trees
        estimated_locations = self._wedge_matching(self.wedges, current_pose)

        if not estimated_locations:
            raise ValueError("No pose estimates found")

        # Return the pose with the highest confidence
        return max(estimated_locations, key=lambda pe: pe.confidence).pose

    def _get_area_of_interest(self, current_pose: Pose2d) -> list[Tree]:
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

    def _create_wedge(self, current_pose: Pose2d, theta: float) -> Wedge:
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

    def _wedge_matching(self, wedges: list[Wedge], current_pose: Pose2d) -> list[PoseEstimate]:
        """Estimate poses by evaluating all skip-allowed wedge→tree maps (length ≥ 2).

        :param wedges: List of candidate Wedge objects.
        :param current_pose: Current estimated pose of the vehicle.
        :return: List of PoseEstimate objects (pose + confidence).
        """
        # Generate maps: {Wedge -> Tree}, allowing skips, with length >= 2
        wedge_combinations = TreeMatcher._generate_wedge_combinations(
            wedges, n=len(wedges), min_len=2
        )

        pose_estimates: list[PoseEstimate] = []
        base_yaw_deg = current_pose.yaw

        # TODO: remove the debug visualizer when done
        debug_visualizer = DebugVisualizer()
        combo_index = 0

        for wedge_map in wedge_combinations:
            # Build line set for this combination
            lines: list[tuple[Point, float]] = []
            for wedge, tree in wedge_map.items():
                phi = math.radians(base_yaw_deg + wedge.theta_degrees)
                lines.append((tree, phi))

            solution = TreeMatcher._solve_least_squares_intersection(lines)
            if solution is None:
                continue

            x_hat, y_hat, rms = solution

            # Compute distance to current pose (optional scoring)
            dx = x_hat - current_pose.x
            dy = y_hat - current_pose.y
            dist = math.hypot(dx, dy)

            combo_index += 1

            # Mark matched trees in wedges
            for wedge, tree in wedge_map.items():
                wedge.matched_tree = tree

            # Debug visualization (only if all wedges were used)
            if len(wedge_map) == len(wedges):
                debug_visualizer.plot_wedges(
                    wedges,
                    current_pose,
                    self.aoi_trees,
                    Point(x_hat, y_hat),
                    f"combo_{combo_index}_dist_{dist:.2f}",
                )

            # Create PoseEstimate
            pose = Pose2d(x_hat, y_hat, current_pose.yaw)
            confidence = 1.0  # TODO: replace with actual confidence calculation
            pose_estimates.append(PoseEstimate(pose, confidence))

        return pose_estimates
