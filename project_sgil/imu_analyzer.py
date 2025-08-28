import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class IMUAnalyzer:
    """
    A class for analyzing IMU data and computing position/yaw changes over specified periods.

    This class loads sensor data from a CSV file and provides methods to calculate changes in
    position (x, y) and yaw angle between specified time indices or over time periods.
    """

    def __init__(self, csv_path: str) -> None:
        """
        Initialize the IMU analyzer with data from a CSV file.

        :param csv_path: Path to the CSV file containing IMU sensor data
        :raises FileNotFoundError: If the CSV file cannot be found
        :raises ValueError: If required columns are missing from the dataset
        """
        self.df = pd.read_csv(csv_path)
        self.timestamps = self.df["timestamp"].values

        # Validate required columns exist
        required_cols = [
            "timestamp",
            "imu_orientation_x",
            "imu_orientation_y",
            "imu_orientation_z",
            "imu_orientation_w",
        ]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Convert quaternion to yaw angle (rotation around z-axis).

        :param qx: Quaternion x component
        :param qy: Quaternion y component
        :param qz: Quaternion z component
        :param qw: Quaternion w component
        :return: Yaw angle in radians
        """
        rot = R.from_quat([qx, qy, qz, qw])
        euler = rot.as_euler("xyz", degrees=False)
        return euler[2]

    def _integrate_motion_period(
        self,
        period_df: pd.DataFrame,
        accel_x_col: str,
        accel_y_col: str,
        angular_vel_col: str,
        orient_cols: list[str] | None = None,
    ) -> tuple[float, float, float]:
        """
        Integrate motion over a specific period to get position and yaw changes.

        :param period_df: DataFrame containing the period data to integrate
        :param accel_x_col: Column name for x-axis linear acceleration
        :param accel_y_col: Column name for y-axis linear acceleration
        :param angular_vel_col: Column name for z-axis angular velocity
        :param orient_cols: List of quaternion column names [x, y, z, w], optional
        :return: Tuple of (delta_x, delta_y, delta_yaw) changes over the period
        """
        # Initialize changes
        delta_x, delta_y = 0.0, 0.0

        # Calculate yaw change
        if orient_cols and all(col in period_df.columns for col in orient_cols):
            # Use quaternion orientation for accurate yaw
            start_row = period_df.iloc[0]
            end_row = period_df.iloc[-1]
            start_yaw = self._quaternion_to_yaw(
                start_row[orient_cols[0]],
                start_row[orient_cols[1]],
                start_row[orient_cols[2]],
                start_row[orient_cols[3]],
            )
            end_yaw = self._quaternion_to_yaw(
                end_row[orient_cols[0]],
                end_row[orient_cols[1]],
                end_row[orient_cols[2]],
                end_row[orient_cols[3]],
            )
            delta_yaw = end_yaw - start_yaw
            current_yaw = start_yaw
        else:
            # Integrate angular velocity
            delta_yaw = 0.0
            current_yaw = 0.0

        # Integrate position changes
        for i in range(1, len(period_df)):
            dt = period_df.iloc[i]["timestamp"] - period_df.iloc[i - 1]["timestamp"]

            # Update yaw for coordinate transformation
            if not (orient_cols and all(col in period_df.columns for col in orient_cols)):
                current_yaw += period_df.iloc[i][angular_vel_col] * dt
                delta_yaw += period_df.iloc[i][angular_vel_col] * dt
            else:
                current_yaw = self._quaternion_to_yaw(
                    period_df.iloc[i][orient_cols[0]],
                    period_df.iloc[i][orient_cols[1]],
                    period_df.iloc[i][orient_cols[2]],
                    period_df.iloc[i][orient_cols[3]],
                )

            # Get accelerations (treating as velocity proxy for short periods)
            vel_x_body = (
                period_df.iloc[i][accel_x_col] if pd.notna(period_df.iloc[i][accel_x_col]) else 0.0
            )
            vel_y_body = (
                period_df.iloc[i][accel_y_col] if pd.notna(period_df.iloc[i][accel_y_col]) else 0.0
            )

            # Transform from body frame to world frame
            vel_x_world = vel_x_body * np.cos(current_yaw) - vel_y_body * np.sin(current_yaw)
            vel_y_world = vel_x_body * np.sin(current_yaw) + vel_y_body * np.cos(current_yaw)

            # Integrate position
            delta_x += vel_x_world * dt
            delta_y += vel_y_world * dt

        return delta_x, delta_y, delta_yaw

    def get_position_yaw_change(
        self, start_index: int, period_length: int, sensor_type: str = "main"
    ) -> dict:
        """
        Calculate the change in position and yaw over a specified period.

        :param start_index: Starting row index in the dataset
        :param period_length: Number of rows to analyze from start_index
        :param sensor_type: Either 'main' for main IMU or 'cheap' for cheap IMU
        :return: Dictionary containing position and yaw changes with timing information
        :raises IndexError: If start_index + period_length exceeds dataset length
        :raises ValueError: If sensor_type is not 'main' or 'cheap'
        """
        # Validate inputs
        if start_index < 0 or start_index >= len(self.df):
            raise IndexError("start_index is out of range")

        end_index = start_index + period_length
        if end_index > len(self.df):
            raise IndexError("start_index + period_length exceeds dataset length")

        if sensor_type not in ["main", "cheap"]:
            raise ValueError("sensor_type must be 'main' or 'cheap'")

        # Extract the period data
        period_df = self.df.iloc[start_index:end_index].copy()

        # Configure sensor-specific column names
        if sensor_type == "main":
            accel_x_col = "imu_linear_acceleration_x"
            accel_y_col = "imu_linear_acceleration_y"
            angular_vel_col = "imu_angular_velocity_z"
            orient_cols = [
                "imu_orientation_x",
                "imu_orientation_y",
                "imu_orientation_z",
                "imu_orientation_w",
            ]
        else:  # cheap
            accel_x_col = "cheap_imu_linear_acceleration_x"
            accel_y_col = "cheap_imu_linear_acceleration_y"
            angular_vel_col = "cheap_imu_angular_velocity_z"
            orient_cols = None  # Cheap IMU doesn't have orientation quaternion

        # Calculate changes using integration
        delta_x, delta_y, delta_yaw = self._integrate_motion_period(
            period_df, accel_x_col, accel_y_col, angular_vel_col, orient_cols
        )

        # Calculate timing information
        start_time = period_df.iloc[0]["timestamp"]
        end_time = period_df.iloc[-1]["timestamp"]
        time_duration = end_time - start_time

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_yaw": delta_yaw,
            "delta_yaw_deg": np.degrees(delta_yaw),
            "time_duration": time_duration,
            "start_time": start_time,
            "end_time": end_time,
            "sensor_type": sensor_type,
            "period_length": period_length,
            "start_index": start_index,
        }
