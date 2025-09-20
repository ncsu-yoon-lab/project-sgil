from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class IMUAnalyzer:
    """Yaw-only analyzer for IMU logs with the OLD schema.

    Uses:
      - timestamp
      - imu_orientation_x, imu_orientation_y, imu_orientation_z, imu_orientation_w
      - imu_angular_velocity_z  (for bridging when a quaternion sample is missing)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with an IMU log DataFrame (old headers).

        :param df: DataFrame with old-format columns.
        """
        self.df: pd.DataFrame = df.copy()

        # Coerce numerics we rely on
        for c in [
            "timestamp",
            "imu_orientation_x",
            "imu_orientation_y",
            "imu_orientation_z",
            "imu_orientation_w",
            "imu_angular_velocity_z",
        ]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        # Time (seconds)
        self.t: np.ndarray = self.df["timestamp"].to_numpy(dtype=float)

        # Quaternions + gyro-z
        self.qx: np.ndarray = self.df.get(
            "imu_orientation_x", pd.Series(np.nan, index=self.df.index)
        ).to_numpy(dtype=float)
        self.qy: np.ndarray = self.df.get(
            "imu_orientation_y", pd.Series(np.nan, index=self.df.index)
        ).to_numpy(dtype=float)
        self.qz: np.ndarray = self.df.get(
            "imu_orientation_z", pd.Series(np.nan, index=self.df.index)
        ).to_numpy(dtype=float)
        self.qw: np.ndarray = self.df.get(
            "imu_orientation_w", pd.Series(np.nan, index=self.df.index)
        ).to_numpy(dtype=float)
        self.wz: np.ndarray = self.df.get(
            "imu_angular_velocity_z", pd.Series(0.0, index=self.df.index)
        ).to_numpy(dtype=float)

        # Precompute continuous yaw (radians)
        self._yaw_unwrapped: np.ndarray = self._build_yaw_series_unwrapped()

    def delta_yaw_deg(self, start_index: int, end_index: int) -> float:
        """Compute signed shortest-path Δyaw (degrees) between two rows.

        :param start_index: Start row index (inclusive).
        :param end_index: End row index (must be > start_index).
        :return: Δyaw in degrees in (-180, 180].
        """
        if end_index <= start_index:
            raise IndexError("end_index must be > start_index.")
        if start_index < 0 or end_index >= len(self._yaw_unwrapped):
            raise IndexError("Indices out of range.")

        dyaw_rad = float(self._yaw_unwrapped[end_index] - self._yaw_unwrapped[start_index])
        dyaw_rad = self._normalize_angle(dyaw_rad)
        return float(np.degrees(dyaw_rad))

    # --------------------------
    # Helpers
    # --------------------------

    def _build_yaw_series_unwrapped(self) -> np.ndarray:
        """Continuous yaw (radians) from quaternions; gyro bridges gaps."""
        n = len(self.df)
        yaw = np.zeros(n, dtype=float)

        # First sample (quat if valid, else start at 0 and integrate)
        has_q0 = np.isfinite(self.qx[0]) and np.isfinite(self.qw[0])
        if has_q0:
            rot0 = R.from_quat([self.qx[0], self.qy[0], self.qz[0], self.qw[0]])
            yaw[0] = rot0.as_euler("xyz", degrees=False)[2]
        else:
            yaw[0] = 0.0  # fall back to 0; subsequent rows may integrate gyro

        for i in range(1, n):
            has_quat = np.isfinite(self.qx[i]) and np.isfinite(self.qw[i])
            if has_quat:
                rot = R.from_quat([self.qx[i], self.qy[i], self.qz[i], self.qw[i]])
                yi = rot.as_euler("xyz", degrees=False)[2]
                # Use shortest-path step from previous unwrapped value
                dy = self._normalize_angle(yi - yaw[i - 1])
                yaw[i] = yaw[i - 1] + dy
            else:
                dt = (
                    float(self.t[i] - self.t[i - 1])
                    if np.isfinite(self.t[i]) and np.isfinite(self.t[i - 1])
                    else 0.0
                )
                wz = float(self.wz[i]) if np.isfinite(self.wz[i]) else 0.0
                yaw[i] = yaw[i - 1] + wz * dt

        # Unwrap to remove residual ±π jumps
        return np.unwrap(yaw)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize radians to (-π, π]."""
        return float(np.arctan2(np.sin(angle), np.cos(angle)))
