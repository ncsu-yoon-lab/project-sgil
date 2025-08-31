from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class YawChange:
    """Container for yaw change results."""

    delta_yaw: float  # radians, signed (shortest path)
    delta_yaw_deg: float  # degrees, signed
    start_yaw: float  # radians
    end_yaw: float  # radians


class IMUAnalyzer:
    """Analyze IMU logs and compute position / yaw changes over time windows.

    The dataset is expected to contain columns for timestamps and IMU
    quaternions (for the main IMU), plus angular velocity and linear
    acceleration signals.
    """

    def __init__(self, csv_path: str) -> None:
        """Initialize the IMU analyzer with data from a CSV file.

        :param csv_path: Path to the CSV file containing IMU sensor
            data.
        :raises FileNotFoundError: If the CSV file cannot be found.
        :raises ValueError: If required columns are missing from the
            dataset.
        """
        self.df: pd.DataFrame = pd.read_csv(csv_path)

        required_cols: list[str] = [
            "timestamp",
            "imu_orientation_x",
            "imu_orientation_y",
            "imu_orientation_z",
            "imu_orientation_w",
            "imu_angular_velocity_z",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure numeric types (robust against CSV type inference)
        for c in required_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        # Keep a NumPy view for speed
        self.timestamps: np.ndarray = self.df["timestamp"].to_numpy(dtype=float)

    # ---------------------------------------------------------------------
    # Core math helpers (public for reuse in tests; no nesting per your style)
    # ---------------------------------------------------------------------

    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """Convert a single quaternion (x, y, z, w) to yaw (radians).

        Uses intrinsic XYZ convention where Z corresponds to yaw.

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
        """Convert arrays of quaternions to a yaw (radians) series.

        :param qx: Array of x components.
        :param qy: Array of y components.
        :param qz: Array of z components.
        :param qw: Array of w components.
        :return: Array of yaw angles in radians.
        """
        rots = R.from_quat(np.column_stack([qx, qy, qz, qw]))
        yaws = rots.as_euler("xyz", degrees=False)[:, 2]
        return yaws.astype(float, copy=False)

    def normalize_angle(self, angle: float) -> float:
        """Normalize an angle to (-π, π].

        :param angle: Angle in radians.
        :return: Normalized angle in radians.
        """
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def shortest_yaw_delta(self, start_yaw: float, end_yaw: float) -> float:
        """Compute shortest signed delta between two yaw angles.

        :param start_yaw: Start yaw in radians.
        :param end_yaw: End yaw in radians.
        :return: Signed shortest-path delta in radians.
        """
        return self.normalize_angle(end_yaw - start_yaw)

    # ---------------------------------------------------------------------
    # Period computations
    # ---------------------------------------------------------------------

    def _extract_period(self, start_index: int, period_length: int) -> pd.DataFrame:
        """Slice the dataframe for a period and validate bounds.

        :param start_index: Starting row index (inclusive).
        :param period_length: Number of rows in the window.
        :return: Period dataframe view (copy).
        :raises IndexError: If the window exceeds dataset bounds.
        """
        if start_index < 0 or start_index >= len(self.df):
            raise IndexError("start_index is out of range.")
        end_index = start_index + period_length
        if end_index > len(self.df):
            raise IndexError("start_index + period_length exceeds dataset length.")
        return self.df.iloc[start_index:end_index].copy()

    def compute_yaw_change_from_quat(self, period_df: pd.DataFrame) -> YawChange:
        """Compute yaw change over a window using quaternion orientations.

        This method converts the quaternion series to yaw, applies phase
        unwrapping to avoid 2π jumps, and returns the signed shortest-
        path delta.

        :param period_df: Windowed dataframe containing quaternion
            columns.
        :return: YawChange with start/end yaw and delta (radians and
            degrees).
        """
        qx = period_df["imu_orientation_x"].to_numpy(dtype=float)
        qy = period_df["imu_orientation_y"].to_numpy(dtype=float)
        qz = period_df["imu_orientation_z"].to_numpy(dtype=float)
        qw = period_df["imu_orientation_w"].to_numpy(dtype=float)

        yaw_series = self.quaternions_to_yaw_series(qx, qy, qz, qw)

        # Unwrap to make the series continuous, then take endpoints
        yaw_unwrapped = np.unwrap(yaw_series)
        start_yaw = float(yaw_unwrapped[0])
        end_yaw = float(yaw_unwrapped[-1])

        # Shortest-path delta for reporting
        delta_yaw = self.shortest_yaw_delta(start_yaw, end_yaw)
        return YawChange(
            delta_yaw=delta_yaw,
            delta_yaw_deg=float(np.degrees(delta_yaw)),
            start_yaw=float(yaw_series[0]),
            end_yaw=float(yaw_series[-1]),
        )

    def compute_yaw_change_from_gyro(
        self, period_df: pd.DataFrame, omega_col: str = "imu_angular_velocity_z"
    ) -> YawChange:
        """Compute yaw change by integrating angular velocity about Z.

        :param period_df: Windowed dataframe with 'timestamp' and omega
            column.
        :param omega_col: Column name for angular velocity (rad/s).
        :return: YawChange with integrated delta yaw.
        """
        t = period_df["timestamp"].to_numpy(dtype=float)
        w = period_df[omega_col].to_numpy(dtype=float)

        # dt between samples (len-1). If single sample, delta is 0.
        if len(t) < 2:
            return YawChange(0.0, 0.0, 0.0, 0.0)

        dt = np.diff(t)
        # Midpoint (rectangular) integration using current sample
        # (trapezoidal would be np.trapz(w, t), but rectangular is fine/consistent)
        delta_yaw = float(np.sum(w[1:] * dt))

        return YawChange(
            delta_yaw=delta_yaw,
            delta_yaw_deg=float(np.degrees(delta_yaw)),
            start_yaw=0.0,
            end_yaw=delta_yaw,
        )

    def _integrate_position_world_xy(
        self,
        period_df: pd.DataFrame,
        accel_x_col: str,
        accel_y_col: str,
        yaw_series: np.ndarray,
    ) -> tuple[float, float]:
        """Very rough integration of body-frame *velocity-like* signals into
        world XY.

        NOTE: This treats linear acceleration columns as if they were body-frame
        velocity samples (a rough proxy for short intervals). For true dead-reckoning,
        you must remove gravity, integrate acceleration→velocity (with bias handling),
        and then velocity→position.

        :param period_df: Window data.
        :param accel_x_col: Column name for x-axis body-frame signal.
        :param accel_y_col: Column name for y-axis body-frame signal.
        :param yaw_series: Yaw (radians) aligned with period_df rows.
        :return: (delta_x, delta_y) in world frame.
        """
        t = period_df["timestamp"].to_numpy(dtype=float)
        if len(t) < 2:
            return 0.0, 0.0

        dt = np.diff(t)

        # Body-frame samples (interpreted here as velocity-like for short windows)
        vx_b = period_df[accel_x_col].to_numpy(dtype=float)
        vy_b = period_df[accel_y_col].to_numpy(dtype=float)

        # World-frame rotation per sample (use current yaw for each step)
        c = np.cos(yaw_series)
        s = np.sin(yaw_series)

        # Rotate body to world for samples 1..N-1 to pair with dt
        vx_w = vx_b * c - vy_b * s
        vy_w = vx_b * s + vy_b * c

        dx = float(np.sum(vx_w[1:] * dt))
        dy = float(np.sum(vy_w[1:] * dt))
        return dx, dy

    # ---------------------------------------------------------------------
    # Public API mirroring your original method
    # ---------------------------------------------------------------------

    def get_position_yaw_change(
        self,
        start_index: int,
        period_length: int,
        sensor_type: str = "main",
        prefer_quaternion_yaw: bool = True,
    ) -> dict[str, float | str | int]:
        """Calculate change in position (rough) and yaw over a specified
        period.

        :param start_index: Starting row index in the dataset.
        :param period_length: Number of rows to analyze from start_index
            (exclusive end).
        :param sensor_type: 'main' (uses imu_* columns) or 'cheap' (uses
            cheap_imu_* columns).
        :param prefer_quaternion_yaw: If True and quaternions exist, use
            quaternion yaw; otherwise integrate angular rate about Z.
        :return: Dictionary of position deltas, yaw deltas, window
            timing, and metadata.
        :raises IndexError: If the requested window exceeds dataset
            length.
        :raises ValueError: If sensor_type is invalid.
        """
        if sensor_type not in ("main", "cheap"):
            raise ValueError("sensor_type must be 'main' or 'cheap'.")

        period_df = self._extract_period(start_index, period_length)

        if sensor_type == "main":
            accel_x_col = "imu_linear_acceleration_x"
            accel_y_col = "imu_linear_acceleration_y"
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
            omega_col = "cheap_imu_angular_velocity_z"
            have_quat = False

        # --- Yaw change ---
        if prefer_quaternion_yaw and have_quat:
            yaw_result = self.compute_yaw_change_from_quat(period_df)
            # Build a yaw series (unwrapped) for transforming body->world in position integration
            qx = period_df["imu_orientation_x"].to_numpy(dtype=float)
            qy = period_df["imu_orientation_y"].to_numpy(dtype=float)
            qz = period_df["imu_orientation_z"].to_numpy(dtype=float)
            qw = period_df["imu_orientation_w"].to_numpy(dtype=float)
            yaw_series = np.unwrap(self.quaternions_to_yaw_series(qx, qy, qz, qw))
        else:
            yaw_result = self.compute_yaw_change_from_gyro(period_df, omega_col=omega_col)
            # For body->world rotation during the window, integrate gyro to get a yaw series
            t = period_df["timestamp"].to_numpy(dtype=float)
            if len(t) >= 2:
                dt = np.diff(t)
                w = period_df[omega_col].to_numpy(dtype=float)
                yaw_series = np.zeros_like(t, dtype=float)
                yaw_series[1:] = np.cumsum(w[1:] * dt)  # start at 0
            else:
                yaw_series = np.zeros_like(t, dtype=float)

        # --- Rough position delta (see note in method docstring) ---
        delta_x, delta_y = self._integrate_position_world_xy(
            period_df, accel_x_col, accel_y_col, yaw_series
        )

        # --- Timing metadata ---
        start_time = float(period_df["timestamp"].iloc[0])
        end_time = float(period_df["timestamp"].iloc[-1])
        time_duration = end_time - start_time

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_yaw": yaw_result.delta_yaw,
            "delta_yaw_deg": yaw_result.delta_yaw_deg,
            "time_duration": time_duration,
            "start_time": start_time,
            "end_time": end_time,
            "sensor_type": sensor_type,
            "period_length": period_length,
            "start_index": start_index,
        }
