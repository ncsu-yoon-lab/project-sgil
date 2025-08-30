"""
IMU trajectory integration and visualization script for robotics sensor data analysis.

Provides functions to integrate IMU motion data with chunked processing to reduce drift
accumulation and visualize trajectories from multiple sensor sources.

file: imu_trajectory_analysis.py
author: Your Name
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Convert quaternion to yaw angle (rotation around z-axis).

    :param qx: Quaternion x component
    :param qy: Quaternion y component
    :param qz: Quaternion z component
    :param qw: Quaternion w component
    :return: Yaw angle in radians
    """
    # Create rotation object from quaternion
    rot = R.from_quat([qx, qy, qz, qw])
    # Get Euler angles (roll, pitch, yaw)
    euler = rot.as_euler("xyz", degrees=False)
    return euler[2]  # Return yaw (z-rotation)


def integrate_motion_chunked(
    df: pd.DataFrame,
    dt_col: str,
    vel_yaw_array: tuple[str, str, str],
    orient_array: tuple[str, str, str, str],
    chunk_size: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate velocity data to get position, processing in chunks to reduce drift.

    If orientation quaternion columns are provided, use them for yaw. Otherwise integrate angular
    velocity. Processing in chunks helps reset drift accumulation that occurs during integration.

    :param df: DataFrame containing sensor data
    :param dt_col: Column name for timestamp data
    :param vel_yaw_array: List of column names for x and y vel and yaw
    :param orient_array: List of column names for orient in x, y, z, w
    :param chunk_size: Number of rows per integration chunk to limit drift
    :return: Tuple of (x_positions, y_positions, yaw_angles) arrays
    """

    # Parsing the lists passed
    vel_x_col = vel_yaw_array[0]
    vel_y_col = vel_yaw_array[1]
    yaw_vel_col = vel_yaw_array[2]
    orient_x_col = orient_array[0]
    orient_y_col = orient_array[1]
    orient_z_col = orient_array[2]
    orient_w_col = orient_array[3]

    # Initialize arrays
    x_pos = np.zeros(len(df))
    y_pos = np.zeros(len(df))
    yaw_angles = np.zeros(len(df))

    # Process data in chunks
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()

        # Reset integration variables for this chunk
        if chunk_start == 0:
            x, y, yaw = 0.0, 0.0, 0.0
        else:
            # Continue from previous chunk's end position
            x = x_pos[chunk_start - 1]
            y = y_pos[chunk_start - 1]
            yaw = yaw_angles[chunk_start - 1]

        # Process each row in the chunk
        for i, (idx, row) in enumerate(chunk_df.iterrows()):
            if i == 0 and chunk_start == 0:
                # Initialize first position
                if all(
                    col in df.columns
                    for col in [orient_x_col, orient_y_col, orient_z_col, orient_w_col]
                ):
                    yaw = quaternion_to_yaw(
                        row[orient_x_col], row[orient_y_col], row[orient_z_col], row[orient_w_col]
                    )
                x_pos[idx] = x
                y_pos[idx] = y
                yaw_angles[idx] = yaw
                continue

            # Calculate time step
            if i == 0:
                dt = chunk_df.iloc[1][dt_col] - chunk_df.iloc[0][dt_col]
            else:
                dt = row[dt_col] - chunk_df.iloc[i - 1][dt_col]

            # Update yaw
            if all(
                col in df.columns and col is not None
                for col in [orient_x_col, orient_y_col, orient_z_col, orient_w_col]
            ):
                # Use quaternion for yaw if available
                yaw = quaternion_to_yaw(
                    row[orient_x_col], row[orient_y_col], row[orient_z_col], row[orient_w_col]
                )
            else:
                # Integrate angular velocity
                yaw += row[yaw_vel_col] * dt

            # Get velocities in body frame (assuming IMU data is in body frame)
            vel_x_body = row[vel_x_col] if pd.notna(row[vel_x_col]) else 0
            vel_y_body = row[vel_y_col] if pd.notna(row[vel_y_col]) else 0

            # Transform to world frame
            vel_x_world = vel_x_body * np.cos(yaw) - vel_y_body * np.sin(yaw)
            vel_y_world = vel_x_body * np.sin(yaw) + vel_y_body * np.cos(yaw)

            # Integrate position
            x += vel_x_world * dt
            y += vel_y_world * dt

            # Store results
            x_pos[idx] = x
            y_pos[idx] = y
            yaw_angles[idx] = yaw

    return x_pos, y_pos, yaw_angles


def plot_imu_trajectories(csv_file_path: str, chunk_size: int = 20) -> dict:
    """
    Plot 3DOF trajectories from Zed IMU and Cheap IMU data with drift reduction.

    :param csv_file_path: Path to the CSV file containing IMU sensor data
    :param chunk_size: Number of rows per integration chunk to reduce drift accumulation
    :return: Dictionary containing trajectory arrays for both sensor types
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # For IMU integration, we'll use linear acceleration as velocity proxy
    # Note: This is a simplification - real integration would require proper velocity estimation

    # Zed IMU trajectory using quaternion orientation
    zed_x, zed_y, zed_yaw = integrate_motion_chunked(
        df,
        "timestamp",
        ("imu_linear_acceleration_x", "imu_linear_acceleration_y", "imu_angular_velocity_z"),
        ("imu_orientation_x", "imu_orientation_y", "imu_orientation_z", "imu_orientation_w"),
        chunk_size=chunk_size,
    )

    # Cheap IMU trajectory (no orientation quaternion available)
    cheap_x, cheap_y, cheap_yaw = integrate_motion_chunked(
        df,
        "timestamp",
        (
            "cheap_imu_linear_acceleration_x",
            "cheap_imu_linear_acceleration_y",
            "cheap_imu_angular_velocity_z",
        ),
        (None, None, None, None),
        chunk_size=chunk_size,
    )

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: 2D trajectory comparison
    axes[0, 0].plot(zed_x, zed_y, "b-", linewidth=2, label="Zed IMU", alpha=0.8)
    axes[0, 0].plot(cheap_x, cheap_y, "r-", linewidth=2, label="Cheap IMU", alpha=0.8)
    axes[0, 0].scatter(zed_x[0], zed_y[0], color="blue", s=100, marker="o", label="Start (Zed)")
    axes[0, 0].scatter(
        cheap_x[0], cheap_y[0], color="red", s=100, marker="o", label="Start (Cheap)"
    )
    axes[0, 0].scatter(zed_x[-1], zed_y[-1], color="blue", s=100, marker="s", label="End (Zed)")
    axes[0, 0].scatter(
        cheap_x[-1], cheap_y[-1], color="red", s=100, marker="s", label="End (Cheap)"
    )
    axes[0, 0].set_xlabel("X Position (integrated from acceleration)")
    axes[0, 0].set_ylabel("Y Position (integrated from acceleration)")
    axes[0, 0].set_title("2D Trajectory Comparison")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].axis("equal")

    # Plot 2: Yaw angles over time
    time_rel = df["timestamp"] - df["timestamp"].iloc[0]
    axes[0, 1].plot(time_rel, np.degrees(zed_yaw), "b-", linewidth=2, label="Zed IMU")
    axes[0, 1].plot(time_rel, np.degrees(cheap_yaw), "r-", linewidth=2, label="Cheap IMU")
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Yaw Angle (degrees)")
    axes[0, 1].set_title("Yaw Angle Comparison")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: X position over time
    axes[1, 0].plot(time_rel, zed_x, "b-", linewidth=2, label="Zed IMU")
    axes[1, 0].plot(time_rel, cheap_x, "r-", linewidth=2, label="Cheap IMU")
    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 0].set_ylabel("X Position")
    axes[1, 0].set_title("X Position vs Time")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Y position over time
    axes[1, 1].plot(time_rel, zed_y, "b-", linewidth=2, label="Zed IMU")
    axes[1, 1].plot(time_rel, cheap_y, "r-", linewidth=2, label="Cheap IMU")
    axes[1, 1].set_xlabel("Time (seconds)")
    axes[1, 1].set_ylabel("Y Position")
    axes[1, 1].set_title("Y Position vs Time")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"Integration performed in chunks of {chunk_size} rows")
    print(f"Total data points: {len(df)}")
    print(f"Time duration: {time_rel.iloc[-1]:.3f} seconds")
    print(f"Zed IMU Final Position: ({zed_x[-1]:.3f}, {zed_y[-1]:.3f})")
    print(f"Cheap IMU Final Position: ({cheap_x[-1]:.3f}, {cheap_y[-1]:.3f})")
    print(f"Zed IMU Final Yaw: {np.degrees(zed_yaw[-1]):.1f}°")
    print(f"Cheap IMU Final Yaw: {np.degrees(cheap_yaw[-1]):.1f}°")

    return {
        "zed_x": zed_x,
        "zed_y": zed_y,
        "zed_yaw": zed_yaw,
        "cheap_x": cheap_x,
        "cheap_y": cheap_y,
        "cheap_yaw": cheap_yaw,
        "time": time_rel,
    }


def plot_imu_trajectories_from_df(
    df: pd.DataFrame, chunk_size: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same function but takes a DataFrame directly instead of reading from file.

    :param df: DataFrame containing IMU sensor data
    :param chunk_size: Number of rows per integration chunk to reduce drift accumulation
    :return: Tuple of (x_positions, y_positions, yaw_angles) arrays
    """
    return integrate_motion_chunked(
        df,
        "timestamp",
        ("imu_linear_acceleration_x", "imu_linear_acceleration_y", "imu_angular_velocity_z"),
        (None, None, None, None),
        chunk_size=chunk_size,
    )


def compare_chunk_sizes(csv_file_path: str, chunk_sizes: list[int] | None = None) -> None:
    """
    Compare trajectories with different chunk sizes to analyze drift effects.

    Creates side-by-side plots showing how different chunk sizes affect trajectory estimation for
    both zed IMU and Cheap IMU sensors.

    :param csv_file_path: Path to the CSV file containing IMU sensor data
    :param chunk_sizes: List of chunk sizes to compare, defaults to [10, 20, 50]
    """
    if chunk_sizes is None:
        chunk_sizes = [10, 20, 50]
    df = pd.read_csv(csv_file_path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["blue", "red", "green", "orange", "purple"]

    for i, chunk_size in enumerate(chunk_sizes):
        zed_x, zed_y, zed_yaw = integrate_motion_chunked(
            df,
            "timestamp",
            ("imu_linear_acceleration_x", "imu_linear_acceleration_y", "imu_angular_velocity_z"),
            ("imu_orientation_x", "imu_orientation_y", "imu_orientation_z", "imu_orientation_w"),
            chunk_size=chunk_size,
        )

        color = colors[i % len(colors)]
        axes[0].plot(
            zed_x, zed_y, color=color, linewidth=2, label=f"Zed IMU (chunk={chunk_size})", alpha=0.8
        )

        cheap_x, cheap_y, cheap_yaw = integrate_motion_chunked(
            df,
            "timestamp",
            (
                "cheap_imu_linear_acceleration_x",
                "cheap_imu_linear_acceleration_y",
                "cheap_imu_angular_velocity_z",
            ),
            (None, None, None, None),
            chunk_size=chunk_size,
        )

        axes[1].plot(
            cheap_x,
            cheap_y,
            color=color,
            linewidth=2,
            label=f"Cheap IMU (chunk={chunk_size})",
            alpha=0.8,
        )

    axes[0].set_title("Zed IMU Trajectories - Different Chunk Sizes")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].set_title("Cheap IMU Trajectories - Different Chunk Sizes")
    axes[1].set_xlabel("X Position")
    axes[1].set_ylabel("Y Position")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()
