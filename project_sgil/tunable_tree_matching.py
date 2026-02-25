# automated_sgil_tuner.py

from __future__ import annotations

import ast
import contextlib
import itertools
import math

import pandas as pd
from constants import (
    CAMERA_HEIGHT_M,
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    TREE_LOCATIONS_PATH,
)
from data_structs import Point, Pose2d, Tree, Wedge

from project_sgil.localization.imu_analyzer import IMUAnalyzer
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter
from project_sgil.utils.utils import _segment_intersects_circle, get_relative_angle, normalize_deg

# ---------- Tunable matcher (weights + geometric thresholds) ----------


class TunableTreeMatcher(TreeMatcher):
    def __init__(
        self,
        plot_wedges: bool = False,
        path: str = TREE_LOCATIONS_PATH,
        w_occ: float = 0.6,
        w_rms: float = 0.8,
        w_theta: float = 0.0,
        # geometric params
        tree_radius_m: float = 1.0,
        theta_tol_deg: float = 1.5,
        heading_error_deg: float = 6.0,
        # number-selected weight (now tuned)
        w_num: float = 0.5,
    ) -> None:
        super().__init__(plot_wedges=plot_wedges, path=path)
        self.w_occ = float(w_occ)
        self.w_rms = float(w_rms)
        self.w_theta = float(w_theta)
        self.w_num = float(w_num)

        self.tree_radius_m = float(tree_radius_m)
        self.theta_tol_deg = float(theta_tol_deg)
        self.heading_error_deg = float(heading_error_deg)

    def _create_wedge(self, current_pose: Pose2d, theta: float) -> Wedge:
        wedge = Wedge(theta)
        for tree in self.aoi_trees:
            rel_angle_deg = get_relative_angle(tree, current_pose)
            if abs(rel_angle_deg - theta) < self.heading_error_deg:
                wedge.trees.append(tree)
        return wedge

    def _calculate_pose_estimate_score(
        self,
        wedge_map: dict[Wedge, Tree],
        estimated_pose: Pose2d,
        residual_rms: float,
    ) -> float:
        if not wedge_map:
            return 0.0

        visible_count = 0
        theta_match_count = 0
        total_selected = len(wedge_map)

        for wedge, selected_tree in wedge_map.items():
            occluded = False
            for other_tree in self.aoi_trees:
                if other_tree.id == selected_tree.id:
                    continue
                if _segment_intersects_circle(
                    start=estimated_pose,
                    end=selected_tree,
                    center=other_tree,
                    radius=self.tree_radius_m,
                ):
                    occluded = True
                    break
            if not occluded:
                visible_count += 1

            observed_deg = get_relative_angle(selected_tree, estimated_pose)
            diff = normalize_deg(observed_deg - wedge.theta_degrees)
            if abs(diff) <= self.theta_tol_deg:
                theta_match_count += 1

        visibility_ratio = visible_count / float(total_selected)
        theta_match_ratio = theta_match_count / float(total_selected)

        from project_sgil.constants import SCORE_RMS_SCALE

        if total_selected <= 2:
            rms_component = 0.0
        else:
            rms_score = 1.0 / (1.0 + (residual_rms / max(1e-9, SCORE_RMS_SCALE)))
            rms_component = self.w_rms * rms_score

        return float(
            self.w_occ * visibility_ratio
            + self.w_theta * theta_match_ratio
            + rms_component
            + self.w_num * total_selected
        )


# ---------- Tuning runner ----------


class AutomatedSGILTuner:
    """Grid-search the weights + geometric thresholds and report the best by
    mean SGIL error."""

    def __init__(
        self,
        labeled_csv_path: str = "../dataset/tables/annotations_interactive.csv",
        image_folder: str = IMAGE_FOLDER_PATH,
        data_log_path: str = DATA_LOGGER_PATH,
        origin: tuple[float, float] = ORIGIN,
    ) -> None:
        self.converter: Converter = Converter(origin[0], origin[1])
        self.image_folder: str = image_folder
        self.image_shape = IMAGE_SHAPE
        self.camera_height_m = CAMERA_HEIGHT_M

        self.robot_data_log: pd.DataFrame = pd.read_csv(data_log_path)
        self.labeled_data: pd.DataFrame = pd.read_csv(labeled_csv_path)

        self.imu_analyzer: IMUAnalyzer = IMUAnalyzer(self.robot_data_log)

        self.current_pose: Pose2d | None = None
        self._last_row_index: int | None = None
        self._last_rtk_xy: tuple[float, float] | None = None
        self._correct_pose_latlon: tuple[float, float] | None = None

    def run(self) -> None:
        # === keep your current ranges, add w_num in {0.2, 0.5, 0.8} ===
        occ_vals = [0.6, 0.8, 1.0]
        rms_vals = [0.4, 0.6, 0.8]
        theta_vals = [
            0.0,
            0.2,
            0.4,
        ]

        tree_radius_vals = [0.8, 1.1, 1.4]
        theta_tol_vals = [0.8, 1.1, 1.4]
        heading_vals = [3.0, 4.0, 6.0]

        num_vals = [0.8]  # 0.5, 0.8, 1.0]  # NEW: number-of-selected weight

        total = (
            len(occ_vals)
            * len(rms_vals)
            * len(theta_vals)
            * len(tree_radius_vals)
            * len(theta_tol_vals)
            * len(heading_vals)
            * len(num_vals)
        )
        best = None  # (mean_err, (w_occ, w_rms, w_theta, tr, t_tol, h_err, w_num))

        i = 0
        for w_occ, w_rms, w_theta, tr, t_tol, h_err, w_num in itertools.product(
            occ_vals, rms_vals, theta_vals, tree_radius_vals, theta_tol_vals, heading_vals, num_vals
        ):
            i += 1
            pct = 100.0 * i / total
            print(
                f"Combo {i}/{total} ({pct:.1f}%): "
                f"w_occ={w_occ:.1f}, w_rms={w_rms:.1f}, w_theta={w_theta:.1f}, "
                f"tree_r={tr:.1f}, theta_tol={t_tol:.1f}, heading_err={h_err:.0f}, w={w_num:.1f}"
            )

            mean_err = self._evaluate_combo(
                w_occ=w_occ,
                w_rms=w_rms,
                w_theta=w_theta,
                tree_radius_m=tr,
                theta_tol_deg=t_tol,
                heading_error_deg=h_err,
                w_num=w_num,
            )
            if mean_err is None:
                continue
            if best is None or mean_err < best[0]:
                best = (mean_err, (w_occ, w_rms, w_theta, tr, t_tol, h_err, w_num))

        if best is None:
            print("\nNo combination produced any successful matches.")
            return

        mean_err, (w_occ, w_rms, w_theta, tr, t_tol, h_err, w_num) = best
        print("\nBest settings (by mean SGIL error):")
        print(f"  SCORE_WEIGHT_OCCLUSION     = {w_occ:.1f}")
        print(f"  SCORE_WEIGHT_RMS           = {w_rms:.1f}")
        print(f"  SCORE_WEIGHT_THETA_MATCH   = {w_theta:.1f}")
        print(f"  TREE_RADIUS_M              = {tr:.1f} m")
        print(f"  THETA_MATCHING_TOLERANCE   = {t_tol:.1f} deg")
        print(f"  HEADING_ERROR_DEG          = {h_err:.0f} deg")
        print(f"  NUMBER_SELECTED_WEIGHT     = {w_num:.1f}")
        print(f"Mean SGIL error (m): {mean_err:.3f}")

    def _evaluate_combo(
        self,
        w_occ: float,
        w_rms: float,
        w_theta: float,
        tree_radius_m: float,
        theta_tol_deg: float,
        heading_error_deg: float,
        w_num: float,
    ) -> float | None:
        matcher = TunableTreeMatcher(
            plot_wedges=False,
            path=TREE_LOCATIONS_PATH,
            w_occ=w_occ,
            w_rms=w_rms,
            w_theta=w_theta,
            w_num=w_num,
            tree_radius_m=tree_radius_m,
            theta_tol_deg=theta_tol_deg,
            heading_error_deg=heading_error_deg,
        )

        self.current_pose = None
        self._last_row_index = None
        self._last_rtk_xy = None
        errors: list[float] = []

        for _, row in self.labeled_data.iterrows():
            image_name: str = str(row.get("image_filename", "")).strip()
            if not image_name:
                continue

            if image_name == "image_1746551245_831563501.jpg":
                break

            rtk_pose = self._get_rtk_pose(image_name)
            if rtk_pose is None:
                continue

            row_index = self._row_index_for_image(image_name)
            if row_index is None:
                continue

            if self.current_pose is None:
                self.current_pose = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=rtk_pose.yaw)
                self._last_row_index = row_index
            else:
                if self._last_rtk_xy is not None:
                    self.current_pose.x += rtk_pose.x - self._last_rtk_xy[0]
                    self.current_pose.y += rtk_pose.y - self._last_rtk_xy[1]
                assert self._last_row_index is not None
                dyaw = self.imu_analyzer.delta_yaw_deg(int(self._last_row_index), row_index)
                self.current_pose.yaw += float(dyaw)
                self._last_row_index = row_index
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            pixel_points = self._parse_tree_points(str(row.get("tree_points", "")).strip())
            if not pixel_points:
                continue

            ground_thetas = [self.converter.image_x_to_theta(pt.x) for pt in pixel_points]
            # pose_for_match = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=self.current_pose.yaw)
            pose_for_match = Pose2d(
                x=self.current_pose.x,
                y=self.current_pose.y,
                yaw=self.current_pose.yaw,
            )

            try:
                est_xy: Point = matcher.match_trees(pose_for_match, ground_thetas)
            except Exception:
                continue

            self.current_pose.x = est_xy.x
            self.current_pose.y = est_xy.y

            err_m = self._sgil_error_meters(Pose2d(est_xy.x, est_xy.y, self.current_pose.yaw))
            if math.isfinite(err_m):
                # ---- quit this combo if error too large ----
                if err_m > 15.0:
                    return None
                errors.append(err_m)

        if len(errors) == 0:
            return None
        return float(sum(errors) / len(errors))

    # ---- helpers (unchanged) ----

    def _get_rtk_pose(self, image_name: str) -> Pose2d | None:
        matches = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        row = matches.iloc[0]
        if pd.isna(row.get("rtk_heading")):
            return None

        lat_rtk = float(row["rtk_lat"])
        lon_rtk = float(row["rtk_lon"])
        yaw_deg = self.converter.heading_to_yaw(float(row["rtk_heading"]))
        x, y = self.converter.latlon_to_xy((lat_rtk, lon_rtk))

        self._correct_pose_latlon = (lat_rtk, lon_rtk)
        return Pose2d(x=x, y=y, yaw=yaw_deg)

    def _row_index_for_image(self, image_name: str) -> int | None:
        matches = self.robot_data_log[
            self.robot_data_log["image_filename"].str.contains(image_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        return int(matches.index[0])

    def _sgil_error_meters(self, estimated_pose: Pose2d) -> float:
        if self._correct_pose_latlon is None:
            return float("nan")
        est_latlon = self.converter.xy_to_latlon(Point(estimated_pose.x, estimated_pose.y))
        return float(
            self.converter.haversine(
                est_latlon[0],
                est_latlon[1],
                self._correct_pose_latlon[0],
                self._correct_pose_latlon[1],
            )
        )

    @staticmethod
    def _parse_tree_points(value: str) -> list[Point]:
        points: list[Point] = []
        if not value:
            return points
        up = value.upper()
        if "NO PATH" in up or "NONE" in up or up == "NAN":
            return points
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return points
        if not isinstance(parsed, list | tuple):
            return points
        for it in parsed:
            if isinstance(it, list | tuple) and len(it) == 2:
                with contextlib.suppress(TypeError, ValueError):
                    points.append(Point(float(it[0]), float(it[1])))
        return points


if __name__ == "__main__":
    AutomatedSGILTuner(
        labeled_csv_path="../dataset/tables/annotations_interactive.csv",
        image_folder=IMAGE_FOLDER_PATH,
        data_log_path=DATA_LOGGER_PATH,
        origin=ORIGIN,
    ).run()
