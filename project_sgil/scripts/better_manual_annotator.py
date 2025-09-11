"""Interactive tree annotation + SGIL matching with IMU yaw updates.

Now performs two matches per image:
  - IMU-yaw:   XY = last estimate + RTK ΔXY; yaw += IMU Δyaw
  - RTK-yaw:   XY = same as above; yaw = RTK yaw for this image

Prints SGIL error for both matches before you decide:
  y -> accept/save (uses IMU-yaw estimate to advance state)
  n -> reselect
  esc -> skip
  q -> quit

Resumes where existing CSV left off.
"""

from __future__ import annotations

import csv
import os

import cv2
import pandas as pd

from project_sgil.constants import (
    CAMERA_HEIGHT_M,
    DATA_LOGGER_PATH,
    IMAGE_FOLDER_PATH,
    IMAGE_SHAPE,
    ORIGIN,
    PLOT,
    TREE_LOCATIONS_PATH,
)
from project_sgil.data_structs import Point, Pose2d
from project_sgil.graphics.debug_visualizer import DebugVisualizer
from project_sgil.localization.imu_analyzer import IMUAnalyzer  # yaw-only version (old headers)
from project_sgil.localization.tree_matcher import TreeMatcher
from project_sgil.utils.converter import Converter

# ----------------------- OpenCV helpers -----------------------


def init_window(window_name: str, x: int = 250, y: int = 250) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, x, y)
    cv2.waitKey(1)


def on_mouse_click(event: int, x: int, y: int, flags: int, param: dict) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    points: list[Point] = param["points"]
    img = param["image"]
    window = param["window"]
    label_prefix = param.get("label_prefix", "T")
    pt = Point(float(x), float(y))
    points.append(pt)
    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    label = f"{label_prefix}{len(points)}"
    cv2.putText(img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow(window, img)


def wait_for_enter_or_esc() -> str:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            return "enter"
        if key == 27:
            return "esc"
        if key in (ord("q"), ord("Q")):
            return "q"


def wait_for_yes_no_esc_q() -> str:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("y"), ord("Y")):
            return "y"
        if key in (ord("n"), ord("N")):
            return "n"
        if key == 27:
            return "esc"
        if key in (ord("q"), ord("Q")):
            return "q"


# ----------------------- Main class -----------------------


class InteractiveSGILAnnotator:
    """Interactive tree selection + SGIL matching with RTK-ΔXY + IMU-Δyaw
    propagation, plus RTK-yaw comparison."""

    def __init__(
        self,
        image_folder: str,
        data_log_path: str,
        output_csv_path: str,
        origin: tuple[float, float] = ORIGIN,
    ) -> None:
        self.converter = Converter(origin[0], origin[1])
        self.tree_matcher = TreeMatcher(False, "../" + TREE_LOCATIONS_PATH)
        self.image_folder = image_folder
        self.output_csv_path = output_csv_path
        self.image_shape = IMAGE_SHAPE
        self.camera_height_m = CAMERA_HEIGHT_M

        # Data
        self.robot_data_log: pd.DataFrame = pd.read_csv(data_log_path)
        self.imu_analyzer: IMUAnalyzer = IMUAnalyzer(self.robot_data_log)

        # Sorted images
        self.image_list: list[str] = sorted(
            f for f in os.listdir(self.image_folder) if f.lower().endswith(".jpg")
        )
        self._index: int = 0

        # Pose / state
        self.current_pose: Pose2d | None = None
        self._last_row_index: int | None = None
        self._last_rtk_xy: tuple[float, float] | None = None
        self._correct_pose_latlon: tuple[float, float] | None = None

        # Resume if CSV exists
        self._resume_from_existing_csv()

        if PLOT:
            DebugVisualizer.clear_plots()

        # Ensure CSV header if starting fresh
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        if not os.path.exists(self.output_csv_path) or os.path.getsize(self.output_csv_path) == 0:
            with open(self.output_csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow(
                    ["image_filename", "num_trees", "tree_points", "path_points"]
                )

    # ----------------------- Public -----------------------

    def run(self) -> None:
        if len(self.image_list) == 0:
            print("No .jpg images found.")
            return

        window = "SGIL Interactive"
        init_window(window, x=200, y=200)

        while self._index < len(self.image_list):
            image_name = self.image_list[self._index]
            image_path = os.path.join(self.image_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[warn] Could not load: {image_name} — skipping.")
                self._index += 1
                continue

            rtk_pose = self._get_rtk_pose(image_name)
            if rtk_pose is None:
                print(f"[warn] No RTK pose for {image_name} — skipping.")
                self._index += 1
                continue

            row_idx = self._row_index_for_image(image_name)
            if row_idx is None:
                print(f"[warn] Could not map {image_name} to row index — skipping.")
                self._index += 1
                continue

            # --- Pose propagation (XY via RTK Δ; yaw via IMU Δ) ---
            if self.current_pose is None:
                # First (or resume seed): seed from RTK
                self.current_pose = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=rtk_pose.yaw)
                self._last_row_index = row_idx
                self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)
            else:
                # XY: add RTK delta to last estimated position
                assert self._last_rtk_xy is not None
                dx_rtk = rtk_pose.x - self._last_rtk_xy[0]
                dy_rtk = rtk_pose.y - self._last_rtk_xy[1]
                self.current_pose.x += dx_rtk
                self.current_pose.y += dy_rtk

                # Yaw: add IMU delta yaw from last accepted row -> current
                assert self._last_row_index is not None
                dyaw_deg = float(self.imu_analyzer.delta_yaw_deg(self._last_row_index, row_idx))
                self.current_pose.yaw += dyaw_deg

                # Advance anchors
                self._last_row_index = row_idx
                self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

            # --- Selection loop (allow reselection with 'n') ---
            while True:
                display = image.copy()
                cv2.imshow(window, display)
                cv2.waitKey(1)

                print(f"\nImage: {image_name}")
                print(
                    "Tree selection: left-click trees; press Enter when done, ESC to skip image, Q to quit."
                )
                tree_points: list[Point] = []
                cb_param = {
                    "points": tree_points,
                    "image": display,
                    "window": window,
                    "label_prefix": "T",
                }
                cv2.setMouseCallback(window, on_mouse_click, cb_param)

                action = wait_for_enter_or_esc()
                if action == "q":
                    print("Quitting.")
                    cv2.destroyAllWindows()
                    return
                if action == "esc":
                    cv2.setMouseCallback(window, lambda *a, **k: None)
                    print("Skipped image.")
                    self._index += 1
                    break

                cv2.setMouseCallback(window, lambda *a, **k: None)

                if len(tree_points) == 0:
                    print(
                        "No trees clicked. Press 'n' to reselect, 'esc' to skip, 'q' to quit, or 'y' to accept empty set."
                    )
                    next_action = wait_for_yes_no_esc_q()
                    if next_action == "q":
                        print("Quitting.")
                        cv2.destroyAllWindows()
                        return
                    if next_action == "esc":
                        print("Skipped image.")
                        self._index += 1
                        break
                    if next_action == "n":
                        print("Reselecting...")
                        continue

                # Angles (deg) from pixel x
                thetas = self._ground_thetas_from_pixels(tree_points)

                # Pose A: IMU-yaw (propagated yaw), and rtk XY
                pose_imu_yaw = Pose2d(rtk_pose.x, rtk_pose.y, self.current_pose.yaw)

                # Two candidate input poses:
                pose_imu_yaw = Pose2d(
                    rtk_pose.x, rtk_pose.y, self.current_pose.yaw
                )  # RTK XY + IMU yaw
                pose_rtk_yaw = rtk_pose  # RTK XY + RTK yaw

                est_xy_imu = None
                est_xy_rtk = None
                imu_ok = False
                rtk_ok = False

                # Try IMU-yaw
                try:
                    est_xy_imu = self.tree_matcher.match_trees(pose_imu_yaw, thetas)
                    imu_ok = True
                except Exception as e:
                    print(f"[error] TreeMatcher (RTK XY + IMU yaw) failed: {e}")

                # Try RTK-yaw (independent attempt)
                try:
                    est_xy_rtk = self.tree_matcher.match_trees(pose_rtk_yaw, thetas)
                    rtk_ok = True
                except Exception as e:
                    print(f"[warn] TreeMatcher (RTK pose) failed: {e}")

                # Report what worked
                if imu_ok or rtk_ok:
                    if imu_ok:
                        est_pose_imu = Pose2d(est_xy_imu.x, est_xy_imu.y, pose_imu_yaw.yaw)
                        err_imu = self._sgil_error_meters(est_pose_imu)
                        print(
                            f"[IMU yaw]  yaw={pose_imu_yaw.yaw:.2f}° | Est=({est_xy_imu.x:.2f},{est_xy_imu.y:.2f}) | SGIL err={err_imu:.2f} m"
                        )
                    else:
                        print("[IMU yaw]  match FAILED")

                    if rtk_ok:
                        est_pose_rtk = Pose2d(est_xy_rtk.x, est_xy_rtk.y, pose_rtk_yaw.yaw)
                        err_rtk = self._sgil_error_meters(est_pose_rtk)
                        print(
                            f"[RTK yaw]  yaw={pose_rtk_yaw.yaw:.2f}° | Est=({est_xy_rtk.x:.2f},{est_xy_rtk.y:.2f}) | SGIL err={err_rtk:.2f} m"
                        )
                    else:
                        print("[RTK yaw]  match FAILED")

                    print("Accept?  y=accept & save  |  n=reselec  |  esc=skip  |  q=quit")
                    decision = wait_for_yes_no_esc_q()
                    if decision == "q":
                        print("Quitting.")
                        cv2.destroyAllWindows()
                        return
                    if decision == "esc":
                        print("Skipped image.")
                        self._index += 1
                        break
                    if decision == "n":
                        print("Reselecting...")
                        continue

                    # 'y' — always save trees, regardless of match success
                    self._append_csv_row(image_name, tree_points, "NO PATH")
                    # Note: Not updating self.current_pose.x/y here because inputs used RTK XY by design.
                    # Yaw is maintained elsewhere via IMU deltas.
                    print("Saved and advanced to next image.")
                    self._index += 1
                    break

                else:
                    # Both failed — still allow save-only
                    print("Both IMU-yaw and RTK-yaw matching failed.")
                    print("  y = save trees only (no pose update) and advance")
                    print("  n = reselect trees")
                    print("  esc = skip image")
                    print("  q = quit")
                    decision = wait_for_yes_no_esc_q()
                    if decision == "q":
                        print("Quitting.")
                        cv2.destroyAllWindows()
                        return
                    if decision == "esc":
                        print("Skipped image.")
                        self._index += 1
                        break
                    if decision == "n":
                        print("Reselecting...")
                        continue

                    # 'y' => save-only
                    self._append_csv_row(image_name, tree_points, "NO PATH")
                    print("Saved trees (no estimate); advancing to next image.")
                    self._index += 1
                    break

        cv2.destroyAllWindows()
        print(f"\nAll done! Saved annotations to: {self.output_csv_path}")

    # ----------------------- Helpers -----------------------

    def _resume_from_existing_csv(self) -> None:
        """Resume after last annotated image; seed anchors from its RTK
        pose."""
        if not os.path.exists(self.output_csv_path) or os.path.getsize(self.output_csv_path) == 0:
            return

        try:
            done = pd.read_csv(self.output_csv_path)
        except Exception:
            return

        if "image_filename" not in done.columns or len(done) == 0:
            return

        last_image = str(done["image_filename"].iloc[-1]).strip()

        if last_image in self.image_list:
            self._index = self.image_list.index(last_image) + 1
        else:
            for idx, name in enumerate(self.image_list):
                if name.endswith(os.path.basename(last_image)):
                    self._index = idx + 1
                    break

        rtk_pose = self._get_rtk_pose(last_image)
        row_idx = self._row_index_for_image(last_image)
        if rtk_pose is not None and row_idx is not None:
            self.current_pose = Pose2d(x=rtk_pose.x, y=rtk_pose.y, yaw=rtk_pose.yaw)
            self._last_row_index = row_idx
            self._last_rtk_xy = (rtk_pose.x, rtk_pose.y)

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

    def _ground_thetas_from_pixels(self, pixel_points: list[Point]) -> list[float]:
        return [self.converter.image_x_to_theta(pt.x) for pt in pixel_points]

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

    def _append_csv_row(
        self, image_name: str, tree_points: list[Point], path_points_csv: str = "NO PATH"
    ) -> None:
        row = [
            image_name,
            len(tree_points),
            [(int(p.x), int(p.y)) for p in tree_points],
            path_points_csv,
        ]
        with open(self.output_csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(row)


# ----------------------- Entrypoint -----------------------

if __name__ == "__main__":
    annotator = InteractiveSGILAnnotator(
        image_folder="../" + IMAGE_FOLDER_PATH,
        data_log_path="../" + DATA_LOGGER_PATH,
        output_csv_path="annotations/annotations_interactive.csv",
        origin=ORIGIN,
    )
    annotator.run()
