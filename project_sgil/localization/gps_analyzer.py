from __future__ import annotations

import math
from collections import deque

import numpy as np


class GPSAnalyzer:
    """Streaming GPS heading estimator.

    Call :meth:`update` with the latest fix (timestamp + lat/lon). The analyzer keeps
    a sliding window of recent *bearings between consecutive fixes* and returns a
    circular-mean averaged heading.

    Conventions:
      - Heading is compass degrees: 0=N, 90=E (in [0, 360)).
      - Yaw (if you compute it elsewhere) can be derived via Converter.heading_to_yaw.

    Notes:
      - Headings are only computed for segments that exceed ``min_distance_m``.
      - If there isn't enough motion/history, heading returns ``None``.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_distance_m: float = 0.75,
        max_dt_s: float | None = None,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if min_distance_m < 0:
            raise ValueError("min_distance_m must be >= 0")

        self.window_size = int(window_size)
        self.min_distance_m = float(min_distance_m)
        self.max_dt_s = float(max_dt_s) if max_dt_s is not None else None

        # Fix history
        self._fixes: deque[tuple[float, float, float]] = deque(maxlen=self.window_size)

        # Segment headings (radians) + weights (meters). Keep one fewer than fixes.
        self._headings_rad: deque[float] = deque(maxlen=max(1, self.window_size - 1))
        self._weights: deque[float] = deque(maxlen=max(1, self.window_size - 1))

    def update(self, timestamp: float, lat: float, lon: float) -> None:
        """Add the newest GPS fix.

        :param timestamp: seconds (or any monotonic unit). Used only for optional filtering.
        :param lat: latitude in degrees.
        :param lon: longitude in degrees.
        """
        if not (math.isfinite(timestamp) and math.isfinite(lat) and math.isfinite(lon)):
            return

        if self._fixes:
            t0, lat0, lon0 = self._fixes[-1]
            dt = float(timestamp - t0)
            if self.max_dt_s is not None and dt > self.max_dt_s:
                # Large time gap: reset history because the segment bearing likely isn't meaningful.
                self.reset()
            elif dt < 0:
                # Out-of-order sample: ignore to preserve causality.
                return

            dist_m = self._haversine_m(lat0, lon0, lat, lon)
            if dist_m >= self.min_distance_m:
                brng_rad = self._initial_bearing_rad(lat0, lon0, lat, lon)
                self._headings_rad.append(brng_rad)
                self._weights.append(dist_m)

        self._fixes.append((float(timestamp), float(lat), float(lon)))

    def reset(self) -> None:
        """Clear all internal history."""
        self._fixes.clear()
        self._headings_rad.clear()
        self._weights.clear()

    def averaged_heading_deg(self) -> float | None:
        """Return the circular-mean averaged heading in degrees (0..360)."""
        if not self._headings_rad:
            return None

        angles = np.asarray(self._headings_rad, dtype=float)
        weights = np.asarray(self._weights, dtype=float) if self._weights else None

        if weights is None or len(weights) != len(angles) or not np.all(np.isfinite(weights)):
            weights = np.ones_like(angles)

        # Weighted circular mean: mean of unit vectors
        s = float(np.sum(weights * np.sin(angles)))
        c = float(np.sum(weights * np.cos(angles)))
        if s == 0.0 and c == 0.0:
            return None

        mean_rad = math.atan2(s, c)
        mean_deg = (math.degrees(mean_rad) + 360.0) % 360.0
        return float(mean_deg)

    # -----------------
    # Geo helpers
    # -----------------

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Minimal local copy (keeps this module self-contained)
        r = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1)
            * math.cos(phi2)
            * math.sin(dlambda / 2.0) ** 2
        )
        return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    @staticmethod
    def _initial_bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle initial bearing from (lat1, lon1) to (lat2, lon2)."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlambda = math.radians(lon2 - lon1)

        y = math.sin(dlambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
        return math.atan2(y, x)
