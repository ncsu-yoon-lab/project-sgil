from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------
# Constants
# -----------------
INPUT_CSV = "../../dataset/tables/Calibration.csv"
OUTPUT_CSV = "../../dataset/tables/CalibrationNew.csv"


def main() -> None:
    # Read new-format CSV
    src = pd.read_csv(INPUT_CSV)

    # Ensure required new-format cols exist (minimal set for conversion)
    required_new = [
        "timestamp_sec", "timestamp_nanosec",
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "angular_vel_x", "angular_vel_y", "angular_vel_z",
        "linear_accel_x", "linear_accel_y", "linear_accel_z",
    ]
    missing = [c for c in required_new if c not in src.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # Build single float timestamp: seconds + nanoseconds * 1e-9
    timestamp = (
        pd.to_numeric(src["timestamp_sec"], errors="coerce").astype(float)
        + pd.to_numeric(src["timestamp_nanosec"], errors="coerce").astype(float) * 1e-9
    )

    # Carry over image filename if present; else empty strings
    if "image_filename" in src.columns:
        image_filename = src["image_filename"].astype(str)
    else:
        image_filename = pd.Series([""] * len(src), index=src.index, dtype=str)

    # Map new → old IMU (main) columns
    imu_orientation_x = pd.to_numeric(src["orientation_x"], errors="coerce")
    imu_orientation_y = pd.to_numeric(src["orientation_y"], errors="coerce")
    imu_orientation_z = pd.to_numeric(src["orientation_z"], errors="coerce")
    imu_orientation_w = pd.to_numeric(src["orientation_w"], errors="coerce")

    imu_angular_velocity_x = pd.to_numeric(src["angular_vel_x"], errors="coerce")
    imu_angular_velocity_y = pd.to_numeric(src["angular_vel_y"], errors="coerce")
    imu_angular_velocity_z = pd.to_numeric(src["angular_vel_z"], errors="coerce")

    imu_linear_acceleration_x = pd.to_numeric(src["linear_accel_x"], errors="coerce")
    imu_linear_acceleration_y = pd.to_numeric(src["linear_accel_y"], errors="coerce")
    imu_linear_acceleration_z = pd.to_numeric(src["linear_accel_z"], errors="coerce")

    # Fill out old-format columns that don't exist in the new dataset.
    # Use NaN where truly unknown; zeros where it makes sense (e.g., throttles default to 0).
    n = len(src)
    NaN = np.nan

    # zed_imu_* (old file had zeros/identity; keep neutral defaults)
    zed_imu_orientation_x = pd.Series(np.zeros(n), dtype=float)
    zed_imu_orientation_y = pd.Series(np.zeros(n), dtype=float)
    zed_imu_orientation_z = pd.Series(np.zeros(n), dtype=float)
    zed_imu_orientation_w = pd.Series(np.ones(n), dtype=float)

    zed_imu_angular_velocity_x = pd.Series(np.zeros(n), dtype=float)
    zed_imu_angular_velocity_y = pd.Series(np.zeros(n), dtype=float)
    zed_imu_angular_velocity_z = pd.Series(np.zeros(n), dtype=float)

    zed_imu_linear_acceleration_x = pd.Series(np.zeros(n), dtype=float)
    zed_imu_linear_acceleration_y = pd.Series(np.zeros(n), dtype=float)
    zed_imu_linear_acceleration_z = pd.Series(np.zeros(n), dtype=float)

    # cheap_imu_* (not available → NaN)
    cheap_imu_angular_velocity_x = pd.Series([NaN] * n, dtype=float)
    cheap_imu_angular_velocity_y = pd.Series([NaN] * n, dtype=float)
    cheap_imu_angular_velocity_z = pd.Series([NaN] * n, dtype=float)
    cheap_imu_linear_acceleration_x = pd.Series([NaN] * n, dtype=float)
    cheap_imu_linear_acceleration_y = pd.Series([NaN] * n, dtype=float)
    cheap_imu_linear_acceleration_z = pd.Series([NaN] * n, dtype=float)

    # zed_pose_* (unknown here)
    zed_pose_position_x = pd.Series([NaN] * n, dtype=float)
    zed_pose_position_y = pd.Series([NaN] * n, dtype=float)
    zed_pose_position_z = pd.Series([NaN] * n, dtype=float)
    zed_pose_orientation_x = pd.Series([NaN] * n, dtype=float)
    zed_pose_orientation_y = pd.Series([NaN] * n, dtype=float)
    zed_pose_orientation_z = pd.Series([NaN] * n, dtype=float)
    zed_pose_orientation_w = pd.Series([NaN] * n, dtype=float)

    # RTK/GPS fields (unknown here)
    rtk_lat = pd.Series([NaN] * n, dtype=float)
    rtk_lon = pd.Series([NaN] * n, dtype=float)
    rtk_alt = pd.Series([NaN] * n, dtype=float)
    rtk_heading = pd.Series([NaN] * n, dtype=float)
    rtk_sats = pd.Series([NaN] * n, dtype=float)

    gps_lat = pd.Series([NaN] * n, dtype=float)
    gps_lon = pd.Series([NaN] * n, dtype=float)
    gps_heading = pd.Series([NaN] * n, dtype=float)
    gps_sats = pd.Series([NaN] * n, dtype=float)
    gps_qual = pd.Series([NaN] * n, dtype=float)

    # Throttles (defaults to 0.0)
    left_throttle = pd.Series(np.zeros(n), dtype=float)
    right_throttle = pd.Series(np.zeros(n), dtype=float)

    # Assemble final DataFrame in the exact old column order
    out = pd.DataFrame({
        "timestamp": timestamp,
        "imu_orientation_x": imu_orientation_x,
        "imu_orientation_y": imu_orientation_y,
        "imu_orientation_z": imu_orientation_z,
        "imu_orientation_w": imu_orientation_w,
        "imu_angular_velocity_x": imu_angular_velocity_x,
        "imu_angular_velocity_y": imu_angular_velocity_y,
        "imu_angular_velocity_z": imu_angular_velocity_z,
        "imu_linear_acceleration_x": imu_linear_acceleration_x,
        "imu_linear_acceleration_y": imu_linear_acceleration_y,
        "imu_linear_acceleration_z": imu_linear_acceleration_z,
        "zed_imu_orientation_x": zed_imu_orientation_x,
        "zed_imu_orientation_y": zed_imu_orientation_y,
        "zed_imu_orientation_z": zed_imu_orientation_z,
        "zed_imu_orientation_w": zed_imu_orientation_w,
        "zed_imu_angular_velocity_x": zed_imu_angular_velocity_x,
        "zed_imu_angular_velocity_y": zed_imu_angular_velocity_y,
        "zed_imu_angular_velocity_z": zed_imu_angular_velocity_z,
        "zed_imu_linear_acceleration_x": zed_imu_linear_acceleration_x,
        "zed_imu_linear_acceleration_y": zed_imu_linear_acceleration_y,
        "zed_imu_linear_acceleration_z": zed_imu_linear_acceleration_z,
        "cheap_imu_angular_velocity_x": cheap_imu_angular_velocity_x,
        "cheap_imu_angular_velocity_y": cheap_imu_angular_velocity_y,
        "cheap_imu_angular_velocity_z": cheap_imu_angular_velocity_z,
        "cheap_imu_linear_acceleration_x": cheap_imu_linear_acceleration_x,
        "cheap_imu_linear_acceleration_y": cheap_imu_linear_acceleration_y,
        "cheap_imu_linear_acceleration_z": cheap_imu_linear_acceleration_z,
        "zed_pose_position_x": zed_pose_position_x,
        "zed_pose_position_y": zed_pose_position_y,
        "zed_pose_position_z": zed_pose_position_z,
        "zed_pose_orientation_x": zed_pose_orientation_x,
        "zed_pose_orientation_y": zed_pose_orientation_y,
        "zed_pose_orientation_z": zed_pose_orientation_z,
        "zed_pose_orientation_w": zed_pose_orientation_w,
        "rtk_lat": rtk_lat,
        "rtk_lon": rtk_lon,
        "rtk_alt": rtk_alt,
        "rtk_heading": rtk_heading,
        "rtk_sats": rtk_sats,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "gps_heading": gps_heading,
        "gps_sats": gps_sats,
        "gps_qual": gps_qual,
        "left_throttle": left_throttle,
        "right_throttle": right_throttle,
        "image_filename": image_filename,
    }, columns=[
        "timestamp",
        "imu_orientation_x", "imu_orientation_y", "imu_orientation_z", "imu_orientation_w",
        "imu_angular_velocity_x", "imu_angular_velocity_y", "imu_angular_velocity_z",
        "imu_linear_acceleration_x", "imu_linear_acceleration_y", "imu_linear_acceleration_z",
        "zed_imu_orientation_x", "zed_imu_orientation_y", "zed_imu_orientation_z", "zed_imu_orientation_w",
        "zed_imu_angular_velocity_x", "zed_imu_angular_velocity_y", "zed_imu_angular_velocity_z",
        "zed_imu_linear_acceleration_x", "zed_imu_linear_acceleration_y", "zed_imu_linear_acceleration_z",
        "cheap_imu_angular_velocity_x", "cheap_imu_angular_velocity_y", "cheap_imu_angular_velocity_z",
        "cheap_imu_linear_acceleration_x", "cheap_imu_linear_acceleration_y", "cheap_imu_linear_acceleration_z",
        "zed_pose_position_x", "zed_pose_position_y", "zed_pose_position_z",
        "zed_pose_orientation_x", "zed_pose_orientation_y", "zed_pose_orientation_z", "zed_pose_orientation_w",
        "rtk_lat", "rtk_lon", "rtk_alt", "rtk_heading", "rtk_sats",
        "gps_lat", "gps_lon", "gps_heading", "gps_sats", "gps_qual",
        "left_throttle", "right_throttle",
        "image_filename",
    ])

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Converted to old format → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
