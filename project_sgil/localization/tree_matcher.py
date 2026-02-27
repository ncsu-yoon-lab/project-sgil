"""Tree Matcher script to match trees based on the ground position and the
angle of the trees seen from the ground.

file: tree_matcher.py
author: Cole Malinchock and Jack Elia
"""

import collections
import csv
import math

from project_sgil.constants import (
    AOI_ANGLE_DEG,
    AOI_RADIUS_M,
    HEADING_ERROR_DEG,
    HEADING_SWEEP_ENABLED,
    HEADING_SWEEP_RANGE_DEG,
    HEADING_SWEEP_STEP_DEG,
    NUMBER_SELECTED_WEIGHT,
    ORIGIN,
    SCORE_RMS_SCALE,
    SCORE_WEIGHT_OCCLUSION,
    SCORE_WEIGHT_RMS,
    SCORE_WEIGHT_THETA_MATCH,
    THETA_MATCHING_TOLERANCE,
    TREE_LOCATIONS_PATH,
    TREE_RADIUS_M,
    PLOT_WEDGES_PARTIAL,
    HEADING_SCORE_W_DELTA_YAW,
    HEADING_SCORE_W_NUM_WEDGES,
    HEADING_SCORE_W_THETA_ERROR,
)
from project_sgil.data_structs import Point, Pose2d, PoseEstimate, Tree, Wedge
from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.utils.converter import Converter
from project_sgil.utils.utils import _segment_intersects_circle, distance, get_relative_angle, normalize_deg


class TreeMatcher:
    """Match trees based on satellite and ground view data to estimate vehicle
    position."""

    def __init__(self, plot_wedges: bool = False, path: str = TREE_LOCATIONS_PATH) -> None:
        """Initialize the TreeMatcher.

        :param plot_wedges: If True, generate debug plots of wedge
            matching.
        :param path: Path to the CSV file containing tree locations.
        """
        self.plot_wedges = plot_wedges
        self.satellite_tree_locations: list[Point] = []
        self.aoi_trees: list[Tree] = []
        self.wedges: list[Wedge] = []
        self.converter = Converter(ORIGIN[0], ORIGIN[1])

        # Load tree locations from CSV and convert to Point instances
        with open(path, newline="") as csvfile:
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
        :return: (x, y, rms_residual) of intersection, or None if lines
            are degenerate.
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

        When HEADING_SWEEP_ENABLED is True the method tries multiple candidate
        headings (current yaw ± HEADING_SWEEP_RANGE_DEG in steps of
        HEADING_SWEEP_STEP_DEG) and returns the pose with the highest score.
        When disabled, behaviour is identical to the original single-heading
        approach.

        :param current_pose: Current position and heading as Pose2d.
        :param ground_thetas: List of camera angles to trees in ground
            view (positive = left, negative = right).
        :return: Estimated location of the vehicle as Point.
        """

        # Build the list of candidate yaw values to evaluate
        if HEADING_SWEEP_ENABLED:
            candidate_yaws: list[float] = []
            steps = int(HEADING_SWEEP_RANGE_DEG / HEADING_SWEEP_STEP_DEG)
            for i in range(-steps, steps + 1):
                candidate_yaws.append(
                    current_pose.yaw + i * HEADING_SWEEP_STEP_DEG
                )
        else:
            candidate_yaws = [current_pose.yaw]

        # During heading sweep we suppress per-combo plotting inside
        # _wedge_matching and instead plot the best combo per candidate
        # yaw afterwards.
        sweep_active = HEADING_SWEEP_ENABLED and len(candidate_yaws) > 1
        saved_plot_wedges = self.plot_wedges
        if sweep_active:
            self.plot_wedges = False

        # Each entry: (PoseEstimate, wedges_snapshot, aoi_snapshot,
        #              candidate_pose)
        all_entries: list[
            tuple[PoseEstimate, list[Wedge], list[Tree], Pose2d]
        ] = []

        for candidate_yaw in candidate_yaws:
            # Clear previous state for this candidate heading
            self.wedges.clear()
            self.aoi_trees.clear()

            candidate_pose = Pose2d(
                current_pose.x, current_pose.y, candidate_yaw
            )

            # The area of interest based on the candidate pose
            self.aoi_trees = self._get_area_of_interest(candidate_pose)

            # Loop through all the thetas and create wedges
            for theta in ground_thetas:
                self.wedges.append(
                    self._create_wedge(candidate_pose, theta)
                )

            # Estimate location by matching wedges to trees
            pose_estimates = self._wedge_matching(
                self.wedges, candidate_pose
            )

            # Snapshot wedges & aoi for later plotting
            wedges_snap = list(self.wedges)
            aoi_snap = list(self.aoi_trees)

            for pe in pose_estimates:
                all_entries.append(
                    (pe, wedges_snap, aoi_snap, candidate_pose)
                )

        # Restore original plot_wedges setting
        self.plot_wedges = saved_plot_wedges

        if not all_entries:
            if len(ground_thetas) < 2:
                print(
                    "[TreeMatcher] No pose estimates: "
                    "need at least 2 wedges for matching."
                )
            else:
                print(
                    "[TreeMatcher] No pose estimates: no valid "
                    "intersections across all candidate headings."
                )
            raise ValueError("No pose estimates found")

        # --- Pick best per candidate yaw & plot -------------------------
        best_per_yaw: dict[float, tuple[PoseEstimate, list[Wedge], list[Tree], Pose2d]] = {}

        if sweep_active:
            # Group entries by candidate yaw and keep only the best pose-score entry per yaw
            by_yaw: dict[float, list[
                tuple[PoseEstimate, list[Wedge], list[Tree], Pose2d]
            ]] = collections.defaultdict(list)
            for entry in all_entries:
                by_yaw[entry[3].yaw].append(entry)

            for cand_yaw, entries in by_yaw.items():
                best_per_yaw[cand_yaw] = max(entries, key=lambda e: e[0].score)

        # Plot wedges for best scored combo for each yaw, including heading score
        if sweep_active and saved_plot_wedges and best_per_yaw:
            for cand_yaw, best in best_per_yaw.items():
                pe, wedges_snap, aoi_snap, cand_pose = best

                heading_score, theta_rms = self._calculate_heading_score(
                    candidate_yaw=cand_yaw,
                    base_yaw=current_pose.yaw,
                    pose_estimate=pe,
                )

                for w, t in pe.wedge_combinations.items():
                    w.matched_tree = t

                dx = pe.pose.x - cand_pose.x
                dy = pe.pose.y - cand_pose.y
                dist = math.hypot(dx, dy)
                delta = normalize_deg(cand_yaw - current_pose.yaw)

                DebugVisualizer.plot_wedges(
                    wedges=wedges_snap,
                    wedge_combination=pe.wedge_combinations,
                    current_pose=cand_pose,
                    aoi_trees=aoi_snap,
                    estimated_location=Point(pe.pose.x, pe.pose.y),
                    save_name=(
                        f"sweep_yaw_{cand_yaw:.1f}"
                        f"_delta_{delta:+.1f}"
                        f"_dist_{dist:.2f}"
                        f"_wedges_{len(pe.wedge_combinations)}"
                        f"_score_{pe.score:.2f}"
                        f"_thetaRMS_{theta_rms:.2f}"
                        f"_hscore_{heading_score:.2f}"
                        f"_conf_{pe.confidence:.2f}"
                    ),
                )

        # --- Final selection -------------------------------------------
        if sweep_active and best_per_yaw:
            scored: list[tuple[float, float, PoseEstimate]] = []
            for cand_yaw, (pe, _w, _a, _cpose) in best_per_yaw.items():
                heading_score, theta_rms = self._calculate_heading_score(
                    candidate_yaw=cand_yaw,
                    base_yaw=current_pose.yaw,
                    pose_estimate=pe,
                )
                scored.append((heading_score, theta_rms, pe))

            # highest heading_score wins
            heading_score, theta_rms, final_estimate = max(scored, key=lambda x: x[0])

            print(
                f"[TreeMatcher] Heading sweep selected yaw={final_estimate.pose.yaw:.2f} "
                f"(input was {current_pose.yaw:.2f}, delta={normalize_deg(final_estimate.pose.yaw - current_pose.yaw):+.2f}) "
                f"pose_score={final_estimate.score:.2f} theta_rms={theta_rms:.2f} heading_score={heading_score:.2f}"
            )
        else:
            # Find the best pose estimate across all candidate headings
            best_entry = max(all_entries, key=lambda e: e[0].score)
            final_estimate = best_entry[0]

        # Set wedges for visualization
        for wedge, tree in final_estimate.wedge_combinations.items():
            wedge.matched_tree = tree

        # Return the pose with the highest score
        return final_estimate.pose

    def _get_area_of_interest(self, current_pose: Pose2d) -> list[Tree]:
        """Identify satellite trees within area of interest.

        :param current_pose: Current position estimation as Pose2d.
        :return: List of tree locations (Point) within the area of
            interest.
        """

        area_of_interest_tree_locations: list[Tree] = []

        # Computes adjusted pose offset 10m behind current heading
        adjusted_pose = Pose2d(
            x=current_pose.x - 5 * math.cos(math.radians(current_pose.yaw)),
            y=current_pose.y - 5 * math.sin(math.radians(current_pose.yaw)),
            yaw=current_pose.yaw,
        )

        for tree_id, tree in enumerate(self.satellite_tree_locations):
            # Checks if the distance is within the radius
            if distance(tree, adjusted_pose) < AOI_RADIUS_M:
                # Checks if the relative angle to the tree is within the expected limit
                rel_angle_deg = get_relative_angle(tree, adjusted_pose)
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

        return wedge

    def _wedge_matching(self, wedges: list[Wedge], current_pose: Pose2d) -> list[PoseEstimate]:
        """Estimate poses by evaluating all skip-allowed wedge→tree maps
        (length ≥ 2).

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

            estimated_pose = Pose2d(x_hat, y_hat, current_pose.yaw)

            # Compute distance to actual pose
            dx = x_hat - current_pose.x
            dy = y_hat - current_pose.y
            dist = math.hypot(dx, dy)

            combo_index += 1

            # Mark matched trees in wedges
            for wedge, tree in wedge_map.items():
                wedge.matched_tree = tree

            # Create PoseEstimate
            score = self._calculate_pose_estimate_score(
                wedge_map=wedge_map,
                estimated_pose=estimated_pose,
                residual_rms=rms,
            )

            confidence = self.calculate_pose_estimate_confidence(
                PoseEstimate(Pose2d(x_hat, y_hat, current_pose.yaw), score, 1, wedge_map),
                list(wedge_map.values()),
                self.aoi_trees,
            )

            # Debug visualization
            if self.plot_wedges and (
                (PLOT_WEDGES_PARTIAL and len(wedge_map) >= 2)
                or (len(wedge_map) == len(wedges))
            ):
                DebugVisualizer.plot_wedges(
                    wedges=wedges,
                    wedge_combination=wedge_map,
                    current_pose=current_pose,
                    aoi_trees=self.aoi_trees,
                    estimated_location=Point(x_hat, y_hat),
                    save_name=(
                        f"combo_{combo_index}_dist_{dist:.2f}_wedges_{len(wedge_map)}_"
                        f"score_{score:.2f}_conf_{confidence:.2f}"
                    ),
                )

            pose_estimates.append(PoseEstimate(estimated_pose, score, confidence, wedge_map))

        return pose_estimates

    def _calculate_pose_estimate_score(
        self,
        wedge_map: dict[Wedge, Tree],
        estimated_pose: Pose2d,
        residual_rms: float,
    ) -> float:
        """Heuristic score using number of wedges, occlusion, RMS (if 3+
        trees), and theta matching.

        :param wedge_map: Selected {Wedge -> Tree}.
        :param estimated_pose: Pose used as the viewpoint for checks.
        :param residual_rms: LS perpendicular residual (meters).
        :return: Score (higher is better).
        """
        if not wedge_map:
            return 0.0

        visible_count = 0
        theta_match_count = 0
        total_selected = len(wedge_map)

        # For each selected wedge, check (a) occlusion and (b) theta match
        for wedge, selected_tree in wedge_map.items():
            # --- (a) Occlusion: is the line to the selected tree blocked by another tree? ---
            occluded = False
            # Scan all other trees in the aoi
            for other_tree in self.aoi_trees:
                if other_tree.id == selected_tree.id:
                    continue
                if _segment_intersects_circle(
                    start=estimated_pose,
                    end=selected_tree,
                    center=other_tree,
                    radius=TREE_RADIUS_M,
                ):
                    occluded = True
                    break
            if not occluded:
                visible_count += 1

            # --- (b) Theta matching: is the observed angle close to wedge.theta_degrees? ---
            # Observed angle of this tree relative to the estimated pose heading
            observed_deg = get_relative_angle(selected_tree, estimated_pose)

            # Smallest signed difference in degrees (wrap at 180)
            diff = normalize_deg(observed_deg - wedge.theta_degrees)
            if abs(diff) <= THETA_MATCHING_TOLERANCE:
                theta_match_count += 1

        # Occlusion component: fraction of visible selections
        visibility_ratio = visible_count / float(total_selected)
        occlusion_component = SCORE_WEIGHT_OCCLUSION * visibility_ratio

        # TODO: maybe make this not a ratio
        # Theta matching component: fraction of angle-consistent selections
        theta_match_ratio = theta_match_count / float(total_selected)
        theta_match_component = SCORE_WEIGHT_THETA_MATCH * theta_match_ratio

        # RMS component: only count when 3+ trees are used (2 gives trivial RMS≈0)
        if total_selected <= 2:
            rms_component = 0.0
        else:
            rms_score = 1.0 / (1.0 + (residual_rms / max(1e-9, SCORE_RMS_SCALE)))
            rms_component = SCORE_WEIGHT_RMS * rms_score

        # Number of wedges component
        num_wedges_component = total_selected * NUMBER_SELECTED_WEIGHT

        return float(
            occlusion_component + theta_match_component + rms_component + num_wedges_component
        )

    def calculate_pose_estimate_confidence(
        self,
        pose_estimate: PoseEstimate,
        matched_trees: list[Tree],
        all_trees: list[Tree],
    ) -> float:
        """Calculate confidence for a pose estimate.

        Confidence is a heuristic measure of reliability, expressed as a
        value between 0.0 and 1.0. It is independent of the score and
        focuses on the robustness of the match.

        :param pose_estimate: The PoseEstimate to evaluate.
        :param matched_trees: Trees that were successfully matched in
            the estimate.
        :param all_trees: All candidate trees in the area of interest.
        :return: Confidence value in [0.0, 1.0].
        """
        if not all_trees:
            return 0.0

        # Example heuristic: fraction of trees matched
        match_ratio = len(matched_trees) / len(all_trees)

        # Example heuristic: normalize score contribution (optional)
        score_component = min(1.0, pose_estimate.score / 100.0)

        # Blend heuristics (weights can be tuned)
        confidence = 0.7 * match_ratio + 0.3 * score_component

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _calculate_heading_score(
        self,
        *,
        candidate_yaw: float,
        base_yaw: float,
        pose_estimate: PoseEstimate,
    ) -> tuple[float, float]:
        """Post-score heuristic used to select the best yaw during heading sweep.

        This runs *after* `_calculate_pose_estimate_score` has produced a
        best-per-yaw PoseEstimate.

        Components:
          1) Prefer smaller |Δyaw| from the input yaw (closer to 0 deg off).
          2) Prefer higher pose_estimate.score (existing combo score).
          3) Prefer lower total theta error of the matched wedges (RMS in degrees).

        Returns:
          (heading_score, theta_rms_deg)
        """
        # (1) closeness to 0 delta-yaw
        delta = normalize_deg(candidate_yaw - base_yaw)
        delta_score = 1.0 / (1.0 + abs(delta))

        # (3) theta error RMS (lower is better)
        errs: list[float] = []
        for wedge, selected_tree in pose_estimate.wedge_combinations.items():
            observed_deg = get_relative_angle(selected_tree, pose_estimate.pose)
            diff = normalize_deg(observed_deg - wedge.theta_degrees)
            errs.append(diff)

        if errs:
            theta_rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        else:
            theta_rms = float("inf")

        theta_score = 1.0 / (1.0 + theta_rms)

        # (2) number of matched wedges (higher is better) normalized to (0, 1)
        # Use the number of wedges in the *best combo for this yaw*.
        n = float(len(pose_estimate.wedge_combinations))
        num_wedges_score = n / (1.0 + n)

        heading_score = (
            float(HEADING_SCORE_W_DELTA_YAW) * delta_score
            + float(HEADING_SCORE_W_NUM_WEDGES) * num_wedges_score
            + float(HEADING_SCORE_W_THETA_ERROR) * theta_score
        )

        return float(heading_score), float(theta_rms)

