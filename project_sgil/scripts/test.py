from __future__ import annotations

import math
from typing import List, Dict

import numpy as np
import pandas as pd

from imu_analyzer import IMUAnalyzer  # adjust import path if needed

# -----------------
# Constants
# -----------------
CSV_PATH = "Calibration.csv"  # <- set this
AXIS_R = np.eye(3, dtype=float)               # IMU sensor -> world axis map (edit if needed)
SAMPLE_EVERY_SECONDS = 1.0                     # 1 Hz sampling for the printed table
OUTPUT_CSV = "imu_pose_1hz.csv"                # where to save the table


def wrap_deg_0_360(yaw_deg: float) -> float:
    """Wrap yaw in degrees to [0, 360)."""
    yaw_deg = yaw_deg % 360.0
    if yaw_deg < 0.0:
        yaw_deg += 360.0
    return yaw_deg


def main() -> None:
    """Loop through the CSV row-by-row, accumulate pose, and print a 1 Hz table.

    Assumptions:
      - Start pose: (x, y, yaw_deg) = (0, 0, 0).
      - Yaw increases counterclockwise (unit-circle style).
    """
    analyzer = IMUAnalyzer(csv_path=CSV_PATH, axis_R=AXIS_R)

    t = analyzer.t
    n = len(t)
    if n == 0:
        print("Empty CSV.")
        return

    # Pose starts at origin, heading 0°
    x, y, yaw_deg = 0.0, 0.0, 0.0

    rows: List[Dict[str, float]] = []

    # Record the initial state at the exact first timestamp
    rows.append({"timestamp": float(t[0]), "x": x, "y": y, "yaw_deg": yaw_deg})

    # Next whole-second sampling time (at or after the first stamp)
    next_sample = math.floor(float(t[0]))

    # Step through all rows
    for i in range(1, n):
        # Get IMU-only deltas for this step (last -> i)
        dx = analyzer.get_delta_x_since_last(i)
        dy = analyzer.get_delta_y_since_last(i)
        dyaw = analyzer.get_delta_yaw_since_last(i)  # degrees, signed

        # Accumulate
        x += dx
        y += dy
        yaw_deg = wrap_deg_0_360(yaw_deg + dyaw)

        # Emit entries at each whole second boundary crossed up to t[i]
        ti = float(t[i])
        while next_sample + SAMPLE_EVERY_SECONDS <= ti:
            next_sample += SAMPLE_EVERY_SECONDS
            rows.append({"timestamp": next_sample, "x": x, "y": y, "yaw_deg": yaw_deg})

    # If the last timestamp isn't exactly on a second, we've already recorded the last crossed second
    table = pd.DataFrame(rows, columns=["timestamp", "x", "y", "yaw_deg"])

    print("\nPose @ 1 Hz (first 25 rows):")
    print(table.head(100).to_string(index=False))

    table.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
