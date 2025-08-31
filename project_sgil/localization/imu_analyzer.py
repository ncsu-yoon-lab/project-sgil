from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from project_sgil.data_structs import Point

# TODO: this file needs to be rewritten to account for properly handing delta x and y (maybe needs
# to be in a different frame of reference)


@dataclass(frozen=True)
class YawChange:
    """Container for yaw change results.

    :param delta_yaw: Signed shortest-path yaw change in radians.
    :param delta_yaw_deg: Signed shortest-path yaw change in degrees.
    :param start_yaw: Start yaw (radians).
    :param end_yaw: End yaw (radians).
    """

    delta_yaw: float
    delta_yaw_deg: float
    start_yaw: float
    end_yaw: float


class IMUAnalyzer:
    """Analyze IMU logs to compute position and yaw changes over time windows.

    This class rotates body-frame accelerations by the IMU quaternion
    into the world, subtracts gravity, and integrates accel→vel→pos
    (trapezoidal). It auto-detects quaternion convention (body→world vs
    world→body) and supports a fixed extrinsic axis rotation for the IMU
    mounting.
    """

    # --------------------
    # Public
    # --------------------

    def __init__(self, csv_path: str, axis_R: np.ndarray | None = None) -> None:
        """Initialize the analyzer and load data.

        :param csv_path: Path to the CSV file containing IMU sensor
            data.
        :param axis_R: Optional 3x3 rotation matrix to map IMU sensor
            frame to your robot/world frame *before* gravity removal.
            Defaults to identity. Example for a 90° pitch down and Z
            forward, you could build this from Euler.
        :raises ValueError: If required columns are missing or matrices
            are invalid.
        """
        self.df: pd.DataFrame = pd.read_csv(csv_path)

        required_cols: list[str] = [
            "timestamp",
            "imu_orientation_x",
            "imu_orientation_y",
            "imu_orientation_z",
            "imu_orientation_w",
            "imu_angular_velocity_z",
            "imu_linear_acceleration_x",
            "imu_linear_acceleration_y",
            "imu_linear_acceleration_z",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Coerce numeric
        for c in required_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        self.timestamps: np.ndarray = self.df["timestamp"].to_numpy(dtype=float)

        # Fixed sensor->robot/world extrinsic rotation
        if axis_R is None:
            self.axis_R: np.ndarray = np.eye(3, dtype=float)
        else:
            axis_R = np.asarray(axis_R, dtype=float)
            if axis_R.shape != (3, 3):
                raise ValueError("axis_R must be 3x3.")
            # Ensure orthonormal (best-effort)
            U, _, Vt = np.linalg.svd(axis_R)
            self.axis_R = U @ Vt

        # Auto-detected quaternion convention flag:
        # True  -> quat maps body->world (R_bw)
        # False -> quat maps world->body (R_wb)
        self._quat_is_body_to_world: bool | None = None

    def get_position_yaw_change(
        self,
        start_index: int,
        period_length: int,
        sensor_type: str = "main",
        prefer_quaternion_yaw: bool = True,
    ) -> dict[str, float | int | str | Point]:
        """Compute Δx, Δy (world) and Δyaw over a window.

        :param start_index: Starting row index (inclusive).
        :param period_length: Number of rows in the window.
        :param sensor_type: 'main' or 'cheap'. Only 'main' supports full
            gravity removal here.
        :param prefer_quaternion_yaw: Use quaternion yaw if available;
            else integrate ωz.
        :return: Dictionary with displacement/yaw deltas and timing
            metadata.
        :raises IndexError: If the window exceeds dataset bounds.
        :raises ValueError: If the sensor_type is invalid.
        """
        if sensor_type not in ("main", "cheap"):
            raise ValueError("sensor_type must be 'main' or 'cheap'.")

        period_df = self._extract_period(start_index, period_length)

        # Column names
        if sensor_type == "main":
            accel_x_col = "imu_linear_acceleration_x"
            accel_y_col = "imu_linear_acceleration_y"
            accel_z_col = "imu_linear_acceleration_z"
            omega_col = "imu_angular_velocity_z"
            have_quat = (
                period_df[
                    [
                        "imu_orientation_x",
                        "imu_orientation_y",
                        "imu_orientation_z",
                        "imu_orientation_w",
                    ]
                ]
                .notna()
                .all(axis=1)
                .any()
            )
        else:
            accel_x_col = "cheap_imu_linear_acceleration_x"
            accel_y_col = "cheap_imu_linear_acceleration_y"
            accel_z_col = "cheap_imu_linear_acceleration_z"
            omega_col = "cheap_imu_angular_velocity_z"
            have_quat = False

        # Yaw delta
        if prefer_quaternion_yaw and have_quat:
            yaw_res = self.compute_yaw_change_from_quat(period_df)
        else:
            yaw_res = self.compute_yaw_change_from_gyro(period_df, omega_col=omega_col)

        # Displacement Δx, Δy
        if sensor_type == "main" and have_quat:
            if self._quat_is_body_to_world is None:
                # Detect once with a representative slice (first 2-3 seconds, up to 2000 rows)
                self._quat_is_body_to_world = self._detect_quat_convention()

            dx, dy = self._displacement_from_accel_with_orientation(
                period_df=period_df,
                accel_x_col=accel_x_col,
                accel_y_col=accel_y_col,
                accel_z_col=accel_z_col,
                quat_is_body_to_world=bool(self._quat_is_body_to_world),
            )
        else:
            dx, dy = 0.0, 0.0

        start_time = float(period_df["timestamp"].iloc[0])
        end_time = float(period_df["timestamp"].iloc[-1])

        return {
            "delta_x": dx,
            "delta_y": dy,
            "delta_xy": Point(dx, dy),
            "delta_yaw": yaw_res.delta_yaw,
            "delta_yaw_deg": yaw_res.delta_yaw_deg,
            "time_duration": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time,
            "sensor_type": sensor_type,
            "period_length": period_length,
            "start_index": start_index,
        }

    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """Convert a quaternion (x, y, z, w) to yaw (radians).

        :param qx: Quaternion x component.
        :param qy: Quaternion y component.
        :param qz: Quaternion z component.
        :param qw: Quaternion w component.
        :return: Yaw angle in radians.
        """
        rot = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler("xyz", degrees=False)
        return float(yaw)

    def quaternions_to_yaw_series(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        qz: np.ndarray,
        qw: np.ndarray,
    ) -> np.ndarray:
        """Convert arrays of quaternions to a yaw series (radians).

        :param qx: Array of x components.
        :param qy: Array of y components.
        :param qz: Array of z components.
        :param qw: Array of w components.
        :return: Array of yaw angles in radians.
        """
        rots = R.from_quat(np.column_stack([qx, qy, qz, qw]))
        yaws = rots.as_euler("xyz", degrees=False)[:, 2]
        return yaws.astype(float, copy=False)

    def compute_yaw_change_from_quat(self, period_df: pd.DataFrame) -> YawChange:
        """Compute yaw change in a window using quaternion orientations.

        Unwraps the yaw series to avoid ±π jumps, then reports shortest-
        path delta.

        :param period_df: Windowed dataframe with quaternion columns.
        :return: YawChange.
        """
        qx = period_df["imu_orientation_x"].to_numpy(dtype=float)
        qy = period_df["imu_orientation_y"].to_numpy(dtype=float)
        qz = period_df["imu_orientation_z"].to_numpy(dtype=float)
        qw = period_df["imu_orientation_w"].to_numpy(dtype=float)

        yaw_series = self.quaternions_to_yaw_series(qx, qy, qz, qw)
        yaw_unwrapped = np.unwrap(yaw_series)

        start_yaw = float(yaw_unwrapped[0])
        end_yaw = float(yaw_unwrapped[-1])
        delta = self._normalize_angle(end_yaw - start_yaw)

        return YawChange(
            delta_yaw=delta,
            delta_yaw_deg=float(np.degrees(delta)),
            start_yaw=float(yaw_series[0]),
            end_yaw=float(yaw_series[-1]),
        )

    def compute_yaw_change_from_gyro(
        self, period_df: pd.DataFrame, omega_col: str = "imu_angular_velocity_z"
    ) -> YawChange:
        """Compute yaw change by integrating angular velocity about Z.

        :param period_df: Windowed dataframe with 'timestamp' and omega
            column.
        :param omega_col: Column name of angular velocity (rad/s).
        :return: YawChange.
        """
        t = period_df["timestamp"].to_numpy(dtype=float)
        w = period_df[omega_col].to_numpy(dtype=float)
        if len(t) < 2:
            return YawChange(0.0, 0.0, 0.0, 0.0)

        delta = float(np.trapz(w, t))
        return YawChange(
            delta_yaw=delta,
            delta_yaw_deg=float(np.degrees(delta)),
            start_yaw=0.0,
            end_yaw=delta,
        )

    # --------------------
    # Private
    # --------------------

    def _extract_period(self, start_index: int, period_length: int) -> pd.DataFrame:
        """Slice the dataframe for a period and validate bounds.

        :param start_index: Starting row index (inclusive).
        :param period_length: Number of rows in the window.
        :return: Period dataframe copy.
        :raises IndexError: If the window exceeds dataset bounds.
        """
        if start_index < 0 or start_index >= len(self.df):
            raise IndexError("start_index is out of range.")
        end_index = start_index + period_length
        if end_index > len(self.df):
            raise IndexError("start_index + period_length exceeds dataset length.")
        return self.df.iloc[start_index:end_index].copy()

    def _detect_quat_convention(self, sample_len: int = 2000) -> bool:
        """Decide whether quaternions map body→world or world→body.

        We compare two hypotheses by rotating sample accelerations and
        measuring the residual of the world Z component vs +g. The lower
        residual/variance wins.

        :param sample_len: How many rows to sample from the start of the
            dataset.
        :return: True if body→world; False if world→body.
        """
        n = min(sample_len, len(self.df))
        if n < 5:
            return True  # default

        sl = self.df.iloc[:n]
        ax_b = sl["imu_linear_acceleration_x"].to_numpy(dtype=float)
        ay_b = sl["imu_linear_acceleration_y"].to_numpy(dtype=float)
        az_b = sl["imu_linear_acceleration_z"].to_numpy(dtype=float)
        qx = sl["imu_orientation_x"].to_numpy(dtype=float)
        qy = sl["imu_orientation_y"].to_numpy(dtype=float)
        qz = sl["imu_orientation_z"].to_numpy(dtype=float)
        qw = sl["imu_orientation_w"].to_numpy(dtype=float)

        rots = R.from_quat(np.column_stack([qx, qy, qz, qw]))
        a_body = np.column_stack([ax_b, ay_b, az_b])

        # Hypothesis A: quats are body→world
        a_w_A = rots.apply(a_body @ self.axis_R.T)  # apply extrinsic axis map first
        # Hypothesis B: quats are world→body (use inverse to map body→world)
        a_w_B = rots.inv().apply(a_body @ self.axis_R.T)

        g = 9.80665
        res_A = np.mean((a_w_A[:, 2] - g) ** 2) + 0.1 * np.var(a_w_A[:, 2])
        res_B = np.mean((a_w_B[:, 2] - g) ** 2) + 0.1 * np.var(a_w_B[:, 2])

        return bool(res_A <= res_B)

    def _displacement_from_accel_with_orientation(
        self,
        period_df: pd.DataFrame,
        accel_x_col: str,
        accel_y_col: str,
        accel_z_col: str,
        quat_is_body_to_world: bool,
    ) -> tuple[float, float]:
        """Compute world-frame Δx, Δy using quaternion rotation, axis map,
        gravity removal, and double integration (trapezoidal).

        :param period_df: Windowed dataframe.
        :param accel_x_col: Body-frame ax column.
        :param accel_y_col: Body-frame ay column.
        :param accel_z_col: Body-frame az column.
        :param quat_is_body_to_world: If True, use R(q) as body→world;
            else use R(q).inv().
        :return: (delta_x, delta_y) in meters.
        """
        t = period_df["timestamp"].to_numpy(dtype=float)
        if len(t) < 2:
            return 0.0, 0.0

        # Body accel
        a_b = np.column_stack(
            [
                period_df[accel_x_col].to_numpy(dtype=float),
                period_df[accel_y_col].to_numpy(dtype=float),
                period_df[accel_z_col].to_numpy(dtype=float),
            ]
        )

        # Apply fixed sensor→robot/world axis mapping first
        a_b_mapped = a_b @ self.axis_R.T

        # Quaternions
        rots = R.from_quat(
            np.column_stack(
                [
                    period_df["imu_orientation_x"].to_numpy(dtype=float),
                    period_df["imu_orientation_y"].to_numpy(dtype=float),
                    period_df["imu_orientation_z"].to_numpy(dtype=float),
                    period_df["imu_orientation_w"].to_numpy(dtype=float),
                ]
            )
        )

        # Body->World rotation
        a_w = rots.apply(a_b_mapped) if quat_is_body_to_world else rots.inv().apply(a_b_mapped)

        # Bias mitigation (median per axis) and gravity removal
        a_w = a_w - np.median(a_w, axis=0, keepdims=True)
        g_vec = np.array([0.0, 0.0, 9.80665], dtype=float)
        a_lin = a_w - g_vec  # world linear accel

        # Integrate
        dt = np.diff(t)

        # accel -> velocity
        v = np.zeros_like(a_lin)
        v[1:, :] = np.cumsum(0.5 * (a_lin[:-1, :] + a_lin[1:, :]) * dt[:, None], axis=0)

        # velocity -> displacement
        p = np.zeros_like(v)
        p[1:, :] = np.cumsum(0.5 * (v[:-1, :] + v[1:, :]) * dt[:, None], axis=0)

        dx = float(p[-1, 0])
        dy = float(p[-1, 1])
        return dx, dy

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to (-π, π].

        :param angle: Angle in radians.
        :return: Normalized angle in radians.
        """
        return float(np.arctan2(np.sin(angle), np.cos(angle)))
