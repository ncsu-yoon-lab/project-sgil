"""Tree Matcher script to match trees based on the ground position and the
angle of the trees seen from the ground.

file: tree_matcher.py
author: Cole Malinchock and Jack Elia
"""

import collections
import csv
import math
import os

from project_sgil.constants import (
    AOI_ANGLE_DEG,
    AOI_RADIUS_M,
    HEADING_ERROR_DEG,
    HEADING_SWEEP_ENABLED,
    HEADING_SWEEP_RANGE_DEG,
    HEADING_SWEEP_STEP_DEG,
    MAX_WEDGE_COMBINATIONS,
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
    PLOT_RANGE,
    WEDGE_PLOT_ONLY_IMAGE,
    PLOT_ONLY_CHOSEN_COMBO,
    HEADING_SCORE_W_DELTA_YAW,
    HEADING_SCORE_W_NUM_WEDGES,
    HEADING_SCORE_W_THETA_ERROR,
    POSITION_SWEEP_RANGE,
    POSITION_SWEEP_STEP_SIZE,
    MAX_ESTIMATE_DIST_FROM_SNAPPED_M,
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

        # Default scoring parameters (used by base TreeMatcher)
        self.tree_radius_m = float(TREE_RADIUS_M)
        self.theta_tol_deg = float(THETA_MATCHING_TOLERANCE)
        self.heading_error_deg = float(HEADING_ERROR_DEG)

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
        max_results: int | None = None,  # hard cap on number of results
    ) -> list[dict[Wedge, Tree]]:
        """Gets a dictionary mapping Wedge -> Tree for all possible
        combinations of selecting up to n trees from the wedges, allowing
        skipping wedges, and ensuring no Tree is used more than once per
        combination.

        Combinations are generated in descending size order (largest first)
        so that, if *max_results* is hit, the most informative combinations
        are kept.

        :param wedges: Ordered list of wedges to traverse.
        :param n: Maximum number of selections to include in a result.
        :param min_len: Minimum number of selections required for a
            result.
        :param max_results: Stop early once this many results are found.
            ``None`` means unlimited.
        :return: List of dictionaries mapping Wedge -> Tree (no repeated
            Tree ids).
        """
        results: list[dict[Wedge, Tree]] = []
        # Deduplicate identical results even if reached via different recursion paths.
        # Key is a frozenset of (wedge_index, tree_id) so the order doesn't affect uniqueness.
        seen_maps: set[frozenset[tuple[int, int]]] = set()

        # Generate in descending size order so the largest (most informative)
        # combinations are explored first and kept if we hit max_results.
        for target_len in range(n, min_len - 1, -1):
            if max_results is not None and len(results) >= max_results:
                break
            TreeMatcher._backtrack_wedge_maps(
                wedges=wedges,
                max_length=target_len,
                min_length=target_len,  # exact length for this pass
                wedge_index=0,
                chosen_pairs=[],
                used_ids=set(),
                results=results,
                seen_maps=seen_maps,
                max_results=max_results,
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
        max_results: int | None = None,
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
        :param max_results: Stop generating once this many results exist.
        """
        # Early exit if we already have enough results.
        if max_results is not None and len(results) >= max_results:
            return

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
            if max_results is not None and len(results) >= max_results:
                return
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
                max_results=max_results,
            )
            # Undo selection to explore alternative branches (classic backtracking).
            chosen_pairs.pop()
            used_ids.remove(tree.id)

        # --- Option 2: skip this wedge entirely (skip-allowed policy) ---
        if max_results is not None and len(results) >= max_results:
            return
        TreeMatcher._backtrack_wedge_maps(
            wedges=wedges,
            max_length=max_length,
            min_length=min_length,
            wedge_index=wedge_index + 1,
            chosen_pairs=chosen_pairs,
            used_ids=used_ids,
            results=results,
            seen_maps=seen_maps,
            max_results=max_results,
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
    
    def _plot_no_pose_debug(
        self,
        *,
        wedges: list[Wedge],
        aoi_trees: list[Tree],
        candidate_pose: Pose2d,
        image_name: str | None,
        reason: str,
        rtk_pose: Pose2d
    ) -> None:
        """Plot wedges even when no pose estimate exists (debugging)."""
        try:
            DebugVisualizer.plot_wedges(
                wedges=wedges,
                wedge_combination={},  # nothing matched
                current_pose=candidate_pose,
                rtk_pose=rtk_pose,
                aoi_trees=aoi_trees,
                estimated_location=Point(candidate_pose.x, candidate_pose.y),
                save_name=(
                    f"NOPOSE_{reason}_"
                    f"img_{image_name[7:-4] if image_name else 'unknown'}_"
                    f"yaw_{candidate_pose.yaw:.1f}_"
                    f"aoi_{len(aoi_trees)}_wedges_{len(wedges)}"
                ),
            )
        except Exception as e:
            print(f"[TreeMatcher] Failed to plot NO_POSE debug: {e}")


    def match_trees(
        self,
        current_pose: Pose2d,
        ground_thetas: list[float],
        rtk_pose: Pose2d,
        image_name: str = None,
        *,
        snap_distance_m: float = 0.0,
    ) -> Pose2d:
        """Match trees based on current position and ground view angles.

        When HEADING_SWEEP_ENABLED is True the method tries multiple candidate
        headings (current yaw ± HEADING_SWEEP_RANGE_DEG in steps of
        HEADING_SWEEP_STEP_DEG) and returns the pose with the highest score.
        When disabled, behaviour is identical to the original single-heading
        approach.

        :param current_pose: Current position and heading as Pose2d.
        :param ground_thetas: List of camera angles to trees in ground
            view (positive = left, negative = right).
        :param image_name: Name of the image for debugging purposes
        :param snap_distance_m: Distance (m) between raw GPS position and the road-snapped
            position used to form current_pose. Used to adapt the position sweep range.
        :return: Estimated location of the vehicle as Point.
        """
        
        # --- Position sweep (along-road) ---------------------------------
        # We assume current_pose.yaw is aligned with the road direction (from RoadMatcher).
        # Adapt sweep range based on how far GPS had to be snapped to the road.
        # Rationale: larger snap distance likely means larger along-road uncertainty.
        base_sweep_m = 10
        k_sweep_per_snap_m = 1  # linear coefficient: +1m sweep range per 1m snap distance
        max_sweep_m = int(POSITION_SWEEP_RANGE)

        # Computed sweep range in meters (integer for range())
        sweep_m = int(round(base_sweep_m + k_sweep_per_snap_m * float(max(0.0, snap_distance_m))))
        if sweep_m < base_sweep_m:
            sweep_m = base_sweep_m
        if sweep_m > max_sweep_m:
            sweep_m = max_sweep_m

        position_offsets_m = list(
            range(-sweep_m, sweep_m + 1, POSITION_SWEEP_STEP_SIZE)
        )

        if image_name is not None:
            try:
                print(
                    f"[TreeMatcher] sweep_m=±{sweep_m} (base={base_sweep_m}, k={k_sweep_per_snap_m}, "
                    f"snap_distance_m={float(snap_distance_m):.2f}, cap={max_sweep_m})"
                )
            except Exception:
                pass

        # Build the list of candidate yaw values to evaluate (heading sweep)
        if HEADING_SWEEP_ENABLED:
            candidate_yaws: list[float] = []
            steps = int(HEADING_SWEEP_RANGE_DEG / HEADING_SWEEP_STEP_DEG)
            for i in range(-steps, steps + 1):
                candidate_yaws.append(current_pose.yaw + i * HEADING_SWEEP_STEP_DEG)
        else:
            candidate_yaws = [current_pose.yaw]

        # During heading sweep we suppress per-combo plotting inside
        # _wedge_matching and instead plot the best combo per candidate
        # yaw afterwards.
        sweep_active = HEADING_SWEEP_ENABLED and len(candidate_yaws) > 1
        saved_plot_wedges = self.plot_wedges
        if sweep_active:
            self.plot_wedges = False

        within_range = True

        # Check if the plot is within a range
        if PLOT_RANGE is not None and image_name is not None:
            img_num = int(image_name[13:-4])  # Extract number from "img_XXX.jpg"
            if not (PLOT_RANGE[0] <= img_num <= PLOT_RANGE[1]):
                within_range = False

        # If set, only plot wedges for a specific frame index.
        if within_range and WEDGE_PLOT_ONLY_IMAGE is not None and image_name is not None:
            try:
                img_num = int(image_name[13:-4])
                if img_num != int(WEDGE_PLOT_ONLY_IMAGE):
                    within_range = False
            except Exception:
                within_range = False

        # Each entry: (PoseEstimate, wedges_snapshot, aoi_snapshot, candidate_pose, offset_m)
        all_entries: list[tuple[PoseEstimate, list[Wedge], list[Tree], Pose2d, float]] = []

        last_fail_snapshot = None  # (wedges_snap, aoi_snap, candidate_pose, offset_m)

        # Sweep position offsets; for each swept pose, perform the existing heading sweep.
        base_yaw_rad = math.radians(current_pose.yaw)
        for offset_m in position_offsets_m:
            base_x = current_pose.x + float(offset_m) * math.cos(base_yaw_rad)
            base_y = current_pose.y + float(offset_m) * math.sin(base_yaw_rad)
            base_pose = Pose2d(base_x, base_y, current_pose.yaw)

            for candidate_yaw in candidate_yaws:
                # Clear previous state for this candidate heading
                self.wedges.clear()
                self.aoi_trees.clear()

                candidate_pose = Pose2d(base_pose.x, base_pose.y, candidate_yaw)

                # The area of interest based on the candidate pose
                self.aoi_trees = self._get_area_of_interest(candidate_pose)

                # Loop through all the thetas and create wedges
                for theta in ground_thetas:
                    self.wedges.append(self._create_wedge(candidate_pose, theta))

                # Estimate location by matching wedges to trees
                pose_estimates = self._wedge_matching(
                    self.wedges, candidate_pose, rtk_pose, image_name=image_name
                )

                # Snapshot wedges & aoi for later plotting
                wedges_snap = list(self.wedges)
                aoi_snap = list(self.aoi_trees)
                last_fail_snapshot = (wedges_snap, aoi_snap, candidate_pose, float(offset_m))

                for pe in pose_estimates:
                    all_entries.append((pe, wedges_snap, aoi_snap, candidate_pose, float(offset_m)))

        # Restore original plot_wedges setting
        self.plot_wedges = saved_plot_wedges
        
        if not all_entries:
            # Only plot if user enabled plotting AND within range gate passed
            if saved_plot_wedges and within_range and last_fail_snapshot is not None:
                wedges_snap, aoi_snap, cand_pose, _offset_m = last_fail_snapshot

                # Pick a reason string that's useful
                if len(ground_thetas) < 2:
                    reason = "LT2THETA"
                else:
                    reason = "NOINTERSECTION"

                self._plot_no_pose_debug(
                    wedges=wedges_snap,
                    aoi_trees=aoi_snap,
                    candidate_pose=cand_pose,
                    image_name=image_name,
                    reason=reason,
                    rtk_pose=rtk_pose
                )

            if len(ground_thetas) < 2:
                print("[TreeMatcher] No pose estimates: need at least 2 wedges for matching.")
            else:
                print("[TreeMatcher] No pose estimates: no valid intersections across all candidate headings.")
            raise ValueError("No pose estimates found")
        
        # --- Pick best per candidate yaw & plot -------------------------
        best_per_key: dict[tuple[float, float], tuple[PoseEstimate, list[Wedge], list[Tree], Pose2d, float]] = {}

        if sweep_active:
            # Group entries by (offset_m, candidate_yaw) and keep only the best pose-score entry per key
            by_key: dict[tuple[float, float], list[tuple[PoseEstimate, list[Wedge], list[Tree], Pose2d, float]]] = collections.defaultdict(list)
            for entry in all_entries:
                pe, w, a, cand_pose, offset_m = entry
                by_key[(offset_m, cand_pose.yaw)].append(entry)

            for key, entries in by_key.items():
                best_per_key[key] = max(entries, key=lambda e: e[0].score)

        # Plot wedges for best scored combo for each yaw, including heading score
        if sweep_active and saved_plot_wedges and best_per_key and within_range:
            for (offset_m, cand_yaw), best in best_per_key.items():
                pe, wedges_snap, aoi_snap, cand_pose, _off = best

                heading_score, theta_rms = self._calculate_heading_score(
                    candidate_yaw=cand_yaw,
                    base_yaw=current_pose.yaw,
                    pose_estimate=pe,
                )

                for w, t in pe.wedge_combinations.items():
                    w.matched_tree = t

                # Distance used for debug naming should be error vs RTK (camera) pose
                dx = pe.pose.x - rtk_pose.x
                dy = pe.pose.y - rtk_pose.y
                dist = math.hypot(dx, dy)
                delta = normalize_deg(cand_yaw - current_pose.yaw)
                
                DebugVisualizer.plot_wedges(
                    wedges=wedges_snap,
                    wedge_combination=pe.wedge_combinations,
                    current_pose=cand_pose,
                    rtk_pose=rtk_pose,
                    aoi_trees=aoi_snap,
                    estimated_location=Point(pe.pose.x, pe.pose.y),
                    save_name=(
                        f"img_name_{image_name[7:-4] if image_name else 'unknown'}"
                        f"_sweep_pos_{offset_m:+.0f}m"
                        f"_sweep_yaw_{cand_yaw:.1f}"
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
        if sweep_active and best_per_key:
            # First: for each position offset, pick the yaw with the highest heading_score.
            best_per_offset: dict[float, tuple[PoseEstimate, float, float, float, Pose2d]] = {}
            # value: (pose_estimate, chosen_yaw, heading_score, theta_rms, chosen_candidate_pose)
            for (offset_m, cand_yaw), (pe, _w, _a, cpose, _off) in best_per_key.items():
                heading_score, theta_rms = self._calculate_heading_score(
                    candidate_yaw=cand_yaw,
                    base_yaw=current_pose.yaw,
                    pose_estimate=pe,
                )
                if offset_m not in best_per_offset or heading_score > best_per_offset[offset_m][2]:
                    best_per_offset[offset_m] = (pe, cand_yaw, heading_score, theta_rms, cpose)

            # Second: choose the best position offset by PoseEstimate.score (your request).
            chosen_offset_m, (final_estimate, chosen_yaw, heading_score, theta_rms, chosen_cpose) = max(
                best_per_offset.items(), key=lambda kv: kv[1][0].score
            )

            # IMPORTANT: PoseEstimate.pose is XY. Ensure returned pose carries the selected yaw.
            final_pose = Pose2d(final_estimate.pose.x, final_estimate.pose.y, chosen_yaw)

            print(
                f"[TreeMatcher] Position sweep selected offset={chosen_offset_m:+.0f}m "
                f"pose_score={final_estimate.score:.2f} "
                f"(chosen_yaw={chosen_yaw:.2f}, input_yaw={current_pose.yaw:.2f}, delta_yaw={normalize_deg(chosen_yaw - current_pose.yaw):+.2f}) "
                f"theta_rms={theta_rms:.2f} heading_score={heading_score:.2f}"
            )

            if PLOT_ONLY_CHOSEN_COMBO and saved_plot_wedges and within_range:
                try:
                    # Reuse the wedges/aoi from the chosen candidate (keyed by offset and yaw)
                    chosen_key = (float(chosen_offset_m), float(chosen_yaw))
                    chosen_entry = best_per_key.get(chosen_key)
                    if chosen_entry is not None:
                        _pe, wedges_snap, aoi_snap, cand_pose, _off = chosen_entry
                    else:
                        # Fallback: plot with the chosen candidate pose but without snapshots
                        wedges_snap, aoi_snap, cand_pose = list(self.wedges), list(self.aoi_trees), chosen_cpose

                    for w, t in final_estimate.wedge_combinations.items():
                        w.matched_tree = t
                    dx = final_pose.x - rtk_pose.x
                    dy = final_pose.y - rtk_pose.y
                    dist = math.hypot(dx, dy)
                    DebugVisualizer.plot_wedges(
                        wedges=wedges_snap,
                        wedge_combination=final_estimate.wedge_combinations,
                        current_pose=cand_pose,
                        rtk_pose=rtk_pose,
                        aoi_trees=aoi_snap,
                        estimated_location=Point(final_pose.x, final_pose.y),
                        save_name=(
                            f"CHOSEN_combo_{getattr(final_estimate, 'combo_index', None)}"
                            f"_img_{image_name[7:-4] if image_name else 'unknown'}"
                            f"_pos_{chosen_offset_m:+.0f}m"
                            f"_dist_{dist:.2f}"
                            f"_wedges_{len(final_estimate.wedge_combinations)}"
                            f"_score_{final_estimate.score:.2f}"
                            f"_conf_{final_estimate.confidence:.2f}"
                        ),
                    )
                except Exception as e:
                    print(f"[TreeMatcher] Failed to plot chosen combo: {e}")
        else:
            # Find the best pose estimate across all candidate headings/offsets
            best_entry = max(all_entries, key=lambda e: e[0].score)
            final_estimate = best_entry[0]
            chosen_offset_m = float(best_entry[4])
            cand_pose = best_entry[3]

            # IMPORTANT: PoseEstimate.pose is XY. Carry yaw from the selected candidate pose.
            final_pose = Pose2d(final_estimate.pose.x, final_estimate.pose.y, cand_pose.yaw)

            if PLOT_ONLY_CHOSEN_COMBO and saved_plot_wedges and within_range:
                try:
                    pe, wedges_snap, aoi_snap, cand_pose, _off = best_entry
                    for w, t in final_estimate.wedge_combinations.items():
                        w.matched_tree = t
                    dx = final_pose.x - rtk_pose.x
                    dy = final_pose.y - rtk_pose.y
                    dist = math.hypot(dx, dy)
                    DebugVisualizer.plot_wedges(
                        wedges=wedges_snap,
                        wedge_combination=final_estimate.wedge_combinations,
                        current_pose=cand_pose,
                        rtk_pose=rtk_pose,
                        aoi_trees=aoi_snap,
                        estimated_location=Point(final_pose.x, final_pose.y),
                        save_name=(
                            f"CHOSEN_combo_{getattr(final_estimate, 'combo_index', None)}"
                            f"_img_{image_name[7:-4] if image_name else 'unknown'}"
                            f"_pos_{chosen_offset_m:+.0f}m"
                            f"_dist_{dist:.2f}"
                            f"_wedges_{len(final_estimate.wedge_combinations)}"
                            f"_score_{final_estimate.score:.2f}"
                            f"_conf_{final_estimate.confidence:.2f}"
                        ),
                    )
                except Exception as e:
                    print(f"[TreeMatcher] Failed to plot chosen combo: {e}")

        # --- Print chosen combo (wedge -> tree) -------------------------
        try:
            if getattr(final_estimate, "combo_index", None) is not None:
                print(
                    f"[TreeMatcher] Chosen combo_index={final_estimate.combo_index} for {image_name if image_name else 'image'}"
                )
            combo_items = sorted(
                final_estimate.wedge_combinations.items(),
                key=lambda wt: wt[0].theta_degrees,
            )
            combo_str = ", ".join(
                f"(θ={w.theta_degrees:+.1f}° -> tree_id={getattr(t,'id',None)} @ ({t.x:.2f},{t.y:.2f}))"
                for w, t in combo_items
            )
            print(
                f"[TreeMatcher] Picked combo for {image_name if image_name else 'image'}: {combo_str}"
            )
        except Exception as e:
            print(f"[TreeMatcher] Failed to print picked combo: {e}")

        # Set wedges for visualization
        for wedge, tree in final_estimate.wedge_combinations.items():
            wedge.matched_tree = tree

        # Return the pose with the highest score
        return final_pose

    def _get_area_of_interest(self, current_pose: Pose2d) -> list[Tree]:
        """Return satellite trees within the AOI wedge in front of the vehicle.

        Mirrors the AOI depiction in DebugVisualizer.plot_aoi(): the AOI is centered
        a bit *behind* the current pose along heading.
        """
        # Mirror DebugVisualizer's AOI center shift (-5m along heading)
        heading_rad = math.radians(current_pose.yaw)
        aoi_center = Pose2d(
            x=current_pose.x - 5.0 * math.cos(heading_rad),
            y=current_pose.y - 5.0 * math.sin(heading_rad),
            yaw=current_pose.yaw,
        )

        aoi_trees: list[Tree] = []
        for i, pt in enumerate(self.satellite_tree_locations):
            dx = pt.x - aoi_center.x
            dy = pt.y - aoi_center.y
            if dx * dx + dy * dy > AOI_RADIUS_M * AOI_RADIUS_M:
                continue

            rel = get_relative_angle(Point(pt.x, pt.y), aoi_center)
            if abs(rel) <= float(AOI_ANGLE_DEG):
                aoi_trees.append(Tree(pt.x, pt.y, id=i))
        return aoi_trees

    def _create_wedge(self, current_pose: Pose2d, theta: float) -> Wedge:
        """Create a wedge (theta ray) and populate candidate trees near that ray."""
        wedge = Wedge(theta)
        for tree in self.aoi_trees:
            rel_angle_deg = get_relative_angle(tree, current_pose)
            if abs(normalize_deg(rel_angle_deg - theta)) < float(self.heading_error_deg):
                wedge.trees.append(tree)
        return wedge

    def _calculate_pose_estimate_score(
        self,
        wedge_map: dict[Wedge, Tree],
        estimated_pose: Pose2d,
        residual_rms: float,
    ) -> float:
        """Heuristic score for a pose estimate. Higher is better."""
        if not wedge_map:
            return 0.0

        total_selected = len(wedge_map)
        visible_count = 0
        theta_match_count = 0

        for wedge, selected_tree in wedge_map.items():
            # Occlusion heuristic: count as visible if no other tree blocks the segment
            occluded = False
            for other_tree in self.aoi_trees:
                if other_tree.id == selected_tree.id:
                    continue
                if _segment_intersects_circle(
                    start=estimated_pose,
                    end=selected_tree,
                    center=other_tree,
                    radius=float(self.tree_radius_m),
                ):
                    occluded = True
                    break
            if not occluded:
                visible_count += 1

            observed_deg = get_relative_angle(selected_tree, estimated_pose)
            diff = normalize_deg(observed_deg - wedge.theta_degrees)
            if abs(diff) <= float(self.theta_tol_deg):
                theta_match_count += 1

        visibility_ratio = visible_count / float(total_selected)
        theta_match_ratio = theta_match_count / float(total_selected)

        # Residual RMS component is only meaningful with enough constraints
        if total_selected <= 2:
            rms_component = 0.0
        else:
            rms_score = 1.0 / (1.0 + (float(residual_rms) / max(1e-9, float(SCORE_RMS_SCALE))))
            rms_component = float(SCORE_WEIGHT_RMS) * rms_score

        return float(
            float(SCORE_WEIGHT_OCCLUSION) * visibility_ratio
            + float(SCORE_WEIGHT_THETA_MATCH) * theta_match_ratio
            + rms_component
            + float(NUMBER_SELECTED_WEIGHT) * float(total_selected)
        )

    def _wedge_matching(
        self,
        wedges: list[Wedge],
        current_pose: Pose2d,
        rtk_pose: Pose2d,
        *,
        image_name: str | None = None,
    ) -> list[PoseEstimate]:
        """Try matching wedges to satellite trees and return PoseEstimates."""
        # Only consider wedges that have at least one candidate
        eligible = [w for w in wedges if w.trees]
        if len(eligible) < 2:
            return []

        combos = self._generate_wedge_combinations(
            wedges=eligible,
            n=len(eligible),
            min_len=2,
            max_results=MAX_WEDGE_COMBINATIONS,
        )

        pose_estimates: list[PoseEstimate] = []

        # Optional plotting gate for partial combos
        do_partial_plot = bool(self.plot_wedges and PLOT_WEDGES_PARTIAL)

        for combo_index, wedge_map in enumerate(combos, start=1):
            # Build lines: each selected tree + its observed bearing implies the vehicle lies on
            # a ray backwards from the tree.
            lines: list[tuple[Point, float]] = []
            for wedge, tree in wedge_map.items():
                # Absolute angle from vehicle to tree is (current_yaw + theta)
                phi = math.radians(current_pose.yaw + wedge.theta_degrees)
                lines.append((Point(tree.x, tree.y), float(phi)))

            sol = self._solve_least_squares_intersection(lines)
            if sol is None:
                continue
            est_x, est_y, rms = sol

            # ---- distance gate vs snapped/current pose ----
            if (
                (est_x - current_pose.x) ** 2 + (est_y - current_pose.y) ** 2
            ) ** 0.5 > MAX_ESTIMATE_DIST_FROM_SNAPPED_M:
                continue

            est_pose = Pose2d(est_x, est_y, current_pose.yaw)

            score = self._calculate_pose_estimate_score(wedge_map, est_pose, rms)
            confidence = max(0.0, min(1.0, score / 200.0))

            pe = PoseEstimate(
                pose=est_pose,
                score=float(score),
                confidence=float(confidence),
                wedge_combinations=wedge_map,
                combo_index=combo_index,
            )
            pose_estimates.append(pe)

            if do_partial_plot and image_name is not None:
                try:
                    for w, t in wedge_map.items():
                        w.matched_tree = t
                    dx = est_x - rtk_pose.x
                    dy = est_y - rtk_pose.y
                    dist = math.hypot(dx, dy)
                    DebugVisualizer.plot_wedges(
                        wedges=eligible,
                        wedge_combination=wedge_map,
                        current_pose=current_pose,
                        rtk_pose=rtk_pose,
                        aoi_trees=self.aoi_trees,
                        estimated_location=Point(est_x, est_y),
                        save_name=(
                            f"combo_{combo_index}_img_{image_name[7:-4]}"
                            f"_dist_{dist:.2f}_wedges_{len(wedge_map)}_score_{score:.2f}_conf_{confidence:.2f}"
                        ),
                    )
                except Exception:
                    pass

        return pose_estimates

    def _calculate_heading_score(
        self,
        *,
        candidate_yaw: float,
        base_yaw: float,
        pose_estimate: PoseEstimate,
    ) -> tuple[float, float]:
        """Return (heading_score, theta_rms).

        heading_score is used only to pick the best yaw for a given position sweep.
        """
        # Penalize yaw change
        delta_yaw = abs(normalize_deg(float(candidate_yaw) - float(base_yaw)))

        # Penalize theta errors for chosen combo by comparing predicted vs wedge theta
        theta_errs: list[float] = []
        est_pose = pose_estimate.pose
        for wedge, tree in pose_estimate.wedge_combinations.items():
            observed_deg = get_relative_angle(tree, est_pose)
            predicted_deg = normalize_deg(float(base_yaw) + float(wedge.theta_degrees))
            err = normalize_deg(observed_deg - predicted_deg)
            theta_errs.append(float(err))

        if theta_errs:
            theta_rms = math.sqrt(sum(e * e for e in theta_errs) / float(len(theta_errs)))
        else:
            theta_rms = 999.0

        num_wedges = float(len(pose_estimate.wedge_combinations))

        heading_score = float(
            -float(HEADING_SCORE_W_DELTA_YAW) * float(delta_yaw)
            + float(HEADING_SCORE_W_NUM_WEDGES) * float(num_wedges)
            - float(HEADING_SCORE_W_THETA_ERROR) * float(theta_rms)
        )
        return heading_score, float(theta_rms)
