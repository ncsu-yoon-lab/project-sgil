from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from project_sgil.data_structs import Point


@dataclass(frozen=True)
class StepCache:
    """Container for the most recent per-step deltas.

    :param index: Index of the current row (the step advanced to).
    :param delta_x: Change in x (meters, world frame) since the previous row.
    :param delta_y: Change in y (meters, world frame) since the previous row.
    :param delta_yaw_deg: Change in yaw (degrees, signed shortest path) since the previous row.
    """
    index: int
    delta_x: float
    delta_y: float
    delta_yaw_deg: float


class IMUAnalyzer:
    """IMU-only per-row deltas (Δx, Δy, Δyaw) for CSV schema:

    Required columns:
      - timestamp_sec, timestamp_nanosec, frame_id, image_filename
      - orientation_x, orientation_y, orientation_z, orientation_w
      - angular_vel_x, angular_vel_y, angular_vel_z
      - linear_accel_x, linear_accel_y, linear_accel_z
      - (covariances present but not used)

    Pipeline per step:
      1) Rotate body-frame acceleration by quaternion into world frame.
      2) Subtract gravity (world +Z).
      3) Trapezoidal integrate accel→velocity, then velocity→position (stateful).
      4) Δyaw from quaternion endpoints (shortest path); gyro fallback if needed.

    Notes:
      - Dead-reckoning drifts without external corrections.
      - If your IMU axes differ from robot/world axes, supply `axis_R` (3×3).
      - Quaternion convention (body→world vs world→body) is auto-detected once.
    """

    def __init__(self, csv_path: str, axis_R: np.ndarray | None = None, g_mps2: float = 9.80665) -> None:
        """Initialize the analyzer from a CSV.

        :param csv_path: Path to the CSV file with the schema above.
        :param axis_R: Optional 3×3 rotation matrix mapping sensor frame → robot/world frame.
        :param g_mps2: Gravity magnitude (m/s^2).
        :raises ValueError: If required columns are missing or `axis_R` has wrong shape.
        """
        self.df: pd.DataFrame = pd.read_csv(csv_path)

        required = [
            "timestamp_sec", "timestamp_nanosec", "frame_id", "image_filename",
            "orientation_x", "orientation_y", "orientation_z", "orientation_w",
            "angular_vel_x", "angular_vel_y", "angular_vel_z",
            "linear_accel_x", "linear_accel_y", "linear_accel_z",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Coerce numerics we need
        num_cols = [
            "timestamp_sec", "timestamp_nanosec",
            "orientation_x", "orientation_y", "orientation_z", "orientation_w",
            "angular_vel_x", "angular_vel_y", "angular_vel_z",
            "linear_accel_x", "linear_accel_y", "linear_accel_z",
        ]
        for c in num_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        # Build float timestamps in seconds
        self.t: np.ndarray = (self.df["timestamp_sec"].to_numpy(dtype=float)
                              + self.df["timestamp_nanosec"].to_numpy(dtype=float) * 1e-9)

        # Cache arrays
        self.ax: np.ndarray = self.df["linear_accel_x"].to_numpy(dtype=float)
        self.ay: np.ndarray = self.df["linear_accel_y"].to_numpy(dtype=float)
        self.az: np.ndarray = self.df["linear_accel_z"].to_numpy(dtype=float)
        self.qx: np.ndarray = self.df["orientation_x"].to_numpy(dtype=float)
        self.qy: np.ndarray = self.df["orientation_y"].to_numpy(dtype=float)
        self.qz: np.ndarray = self.df["orientation_z"].to_numpy(dtype=float)
        self.qw: np.ndarray = self.df["orientation_w"].to_numpy(dtype=float)
        self.wz: np.ndarray = self.df["angular_vel_z"].to_numpy(dtype=float)

        # Optional image names
        self._images: Optional[np.ndarray] = self.df["image_filename"].astype(str).to_numpy()

        # Axis map (sensor → world). Identity by default.
        if axis_R is None:
            self.axis_R: np.ndarray = np.eye(3, dtype=float)
        else:
            axis_R = np.asarray(axis_R, dtype=float)
            if axis_R.shape != (3, 3):
                raise ValueError("axis_R must be 3x3.")
            U, _, Vt = np.linalg.svd(axis_R)   # orthonormalize
            self.axis_R = U @ Vt

        self.g_vec: np.ndarray = np.array([0.0, 0.0, g_mps2], dtype=float)

        # Per-row rotations (carry-forward over NaNs)
        self.rots: List[R] = self._build_rotation_series()

        # Determine quaternion convention once
        self._quat_is_body_to_world: bool = self._detect_quat_convention(sample_len=2000)

        # Precompute world linear acceleration (gravity removed)
        self.a_lin_w: np.ndarray = self._compute_world_linear_accel()

        # Build continuous yaw series (radians) from quats; gyro bridges gaps
        self._yaw_unwrapped: np.ndarray = self._build_yaw_series_unwrapped()

        # Stateful integration
        self._v_w: np.ndarray = np.zeros(3, dtype=float)   # current world velocity
        self._last_index: int = 0                          # last row we advanced from
        self._last_step: Optional[StepCache] = None        # cached last step deltas

    # ---------------------------------------------------------------------
    # Public API: the three per-step delta getters you asked for
    # ---------------------------------------------------------------------

    def get_delta_x_since_last(self, next_index: int) -> float:
        """Get Δx (meters, world frame) since the last row.

        :param next_index: Target row index to advance to (> last index).
        :return: Δx in meters for (last_index → next_index).
        """
        self._advance_to(next_index)
        assert self._last_step is not None
        return self._last_step.delta_x

    def get_delta_y_since_last(self, next_index: int) -> float:
        """Get Δy (meters, world frame) since the last row.

        :param next_index: Target row index to advance to (> last index).
        :return: Δy in meters for (last_index → next_index).
        """
        self._advance_to(next_index)
        assert self._last_step is not None
        return self._last_step.delta_y

    def get_delta_yaw_since_last(self, next_index: int) -> float:
        """Get Δyaw (degrees, signed shortest path) since the last row.

        :param next_index: Target row index to advance to (> last index).
        :return: Δyaw in degrees for (last_index → next_index).
        """
        self._advance_to(next_index)
        assert self._last_step is not None
        return self._last_step.delta_yaw_deg

    # ---------------------------------------------------------------------
    # Private: stepper and helpers
    # ---------------------------------------------------------------------

    def _advance_to(self, next_index: int) -> None:
        """Advance state from the last index to `next_index` exactly once.

        Uses trapezoidal integration over the single step (i0 → i1).

        :param next_index: Row index to advance to (must be > _last_index and < N).
        """
        if next_index == self._last_index and self._last_step is not None:
            return
        if next_index <= self._last_index:
            raise IndexError(f"next_index ({next_index}) must be > last_index ({self._last_index}).")
        if next_index >= len(self.t):
            raise IndexError("next_index out of range.")

        i0 = self._last_index
        i1 = next_index

        dt = float(self.t[i1] - self.t[i0])
        if dt <= 0.0:
            self._last_step = StepCache(index=i1, delta_x=0.0, delta_y=0.0, delta_yaw_deg=0.0)
            self._last_index = i1
            return

        # a -> v (trapezoid)
        a0 = self.a_lin_w[i0]
        a1 = self.a_lin_w[i1]
        v_new = self._v_w + 0.5 * (a0 + a1) * dt

        # v -> p (trapezoid) over this single step
        dx = 0.5 * (self._v_w[0] + v_new[0]) * dt
        dy = 0.5 * (self._v_w[1] + v_new[1]) * dt

        # Update velocity state
        self._v_w = v_new

        # Δyaw from unwrapped yaw series
        dyaw_rad = float(self._yaw_unwrapped[i1] - self._yaw_unwrapped[i0])
        dyaw_rad = self._normalize_angle(dyaw_rad)
        dyaw_deg = float(np.degrees(dyaw_rad))

        self._last_step = StepCache(index=i1, delta_x=float(dx), delta_y=float(dy), delta_yaw_deg=dyaw_deg)
        self._last_index = i1

    def _build_rotation_series(self) -> List[R]:
        """Build per-row rotations, carrying forward last valid quaternion."""
        rots: List[R] = []
        last_rot = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=False)
        for i in range(len(self.t)):
            if np.isfinite(self.qx[i]) and np.isfinite(self.qy[i]) and np.isfinite(self.qz[i]) and np.isfinite(self.qw[i]):
                last_rot = R.from_quat([self.qx[i], self.qy[i], self.qz[i], self.qw[i]])
            rots.append(last_rot)
        return rots

    def _detect_quat_convention(self, sample_len: int = 2000) -> bool:
        """Detect whether quaternions map body→world (True) or world→body (False)."""
        n = min(sample_len, len(self.t))
        if n < 5:
            return True

        a_b = np.column_stack([self.ax[:n], self.ay[:n], self.az[:n]]) @ self.axis_R.T
        a_w_A = np.vstack([self.rots[i].apply(a_b[i]) for i in range(n)])
        a_w_B = np.vstack([self.rots[i].inv().apply(a_b[i]) for i in range(n)])

        zA, zB = a_w_A[:, 2], a_w_B[:, 2]
        g = self.g_vec[2]
        res_A = np.mean((zA - g) ** 2) + 0.1 * np.var(zA)
        res_B = np.mean((zB - g) ** 2) + 0.1 * np.var(zB)
        return bool(res_A <= res_B)

    def _compute_world_linear_accel(self) -> np.ndarray:
        """Rotate body accel to world, remove median bias, subtract gravity."""
        n = len(self.t)
        a_b = np.column_stack([self.ax, self.ay, self.az]) @ self.axis_R.T
        if self._quat_is_body_to_world:
            a_w = np.vstack([self.rots[i].apply(a_b[i]) for i in range(n)])
        else:
            a_w = np.vstack([self.rots[i].inv().apply(a_b[i]) for i in range(n)])

        # Median bias removal (simple, helps a bit over short windows)
        a_w -= np.median(a_w, axis=0, keepdims=True)

        # Remove gravity (world +Z)
        return a_w - self.g_vec[None, :]

    def _build_yaw_series_unwrapped(self) -> np.ndarray:
        """Continuous yaw (radians) from quaternions; gyro bridges missing quat rows."""
        n = len(self.t)
        yaw = np.zeros(n, dtype=float)
        yaw[0] = self.rots[0].as_euler("xyz", degrees=False)[2]

        for i in range(1, n):
            has_prev = np.isfinite(self.qx[i - 1]) and np.isfinite(self.qw[i - 1])
            has_cur = np.isfinite(self.qx[i]) and np.isfinite(self.qw[i])

            if has_prev and has_cur:
                y_prev = self.rots[i - 1].as_euler("xyz", degrees=False)[2]
                y_cur = self.rots[i].as_euler("xyz", degrees=False)[2]
                dy = self._normalize_angle(y_cur - y_prev)
            else:
                dt = float(self.t[i] - self.t[i - 1])
                wz = float(self.wz[i]) if np.isfinite(self.wz[i]) else 0.0
                dy = wz * dt

            yaw[i] = yaw[i - 1] + dy

        return np.unwrap(yaw)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to (-π, π].

        :param angle: Angle in radians.
        :return: Normalized angle in radians.
        """
        return float(np.arctan2(np.sin(angle), np.cos(angle)))
