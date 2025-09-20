from __future__ import annotations

import math

import pandas as pd

# Adjust the import path if your package layout differs
from project_sgil.localization.imu_analyzer import IMUAnalyzer

# -----------------
# Constants
# -----------------
CSV_PATH = "../../dataset/tables/CalibrationNew.csv"  # <- set this to your IMU CSV path
SAMPLE_EVERY_SECONDS = 1.0  # 1 Hz sampling for the printed/saved table
PRINT_ROWS = 50  # how many rows to print to stdout


def wrap_deg_0_360(yaw_deg: float) -> float:
    """Wrap yaw in degrees to [0, 360)."""
    yaw_deg = yaw_deg % 360.0
    if yaw_deg < 0.0:
        yaw_deg += 360.0
    return yaw_deg


def main() -> None:
    # Load CSV and construct analyzer (yaw-only)
    df = pd.read_csv(CSV_PATH)
    analyzer = IMUAnalyzer(df)

    t = analyzer.t
    n = len(t)
    if n == 0:
        print("Empty CSV.")
        return

    # Start heading: 0 deg
    yaw_deg = 0.0

    # Output rows (timestamp, yaw)
    rows: list[dict[str, float]] = []
    rows.append({"timestamp": float(t[0]), "yaw_deg": yaw_deg})

    # Next whole-second sampling time (starting at the first stamp's whole second)
    next_sample = math.floor(float(t[0]))

    # Keep track of last index we accumulated from
    last_index = 0

    # Step through all rows
    for i in range(1, n):
        # Δyaw from last_index -> i (degrees, signed shortest path)
        dyaw = analyzer.delta_yaw_deg(last_index, i)
        yaw_deg = wrap_deg_0_360(yaw_deg + dyaw)

        # Advance
        last_index = i

        # Emit entries at each whole-second boundary crossed up to t[i]
        ti = float(t[i])
        while next_sample + SAMPLE_EVERY_SECONDS <= ti:
            next_sample += SAMPLE_EVERY_SECONDS
            rows.append({"timestamp": next_sample, "yaw_deg": yaw_deg})

    # Build and show table
    table = pd.DataFrame(rows, columns=["timestamp", "yaw_deg"])

    print("\nYaw @ 1 Hz (first rows):")
    print(table.head(PRINT_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
