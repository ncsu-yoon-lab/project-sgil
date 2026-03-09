"""Estimate heading from XY using a first-order filtered stream."""

from __future__ import annotations

from project_sgil.utils.utils import heading_from_dxdy


class HeadingEstimator:
    """Estimate heading from XY positions with a simple low-pass filter."""

    def __init__(self, alpha: float = 0.2, min_movement: float = 0.05) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.min_movement = min_movement
        self._filtered_x: float | None = None
        self._filtered_y: float | None = None
        self._last_heading: float = 0.0

    def update(self, x: float, y: float) -> float:
        """Update the estimator with a new XY fix and return heading in degrees.

        Heading is normalized to [-180, 180). The first update returns 0.0
        until enough movement is observed.
        """
        if self._filtered_x is None or self._filtered_y is None:
            self._filtered_x = float(x)
            self._filtered_y = float(y)
            self._last_heading = 0.0
            return self._last_heading

        prev_x = self._filtered_x
        prev_y = self._filtered_y

        self._filtered_x = self.alpha * float(x) + (1.0 - self.alpha) * prev_x
        self._filtered_y = self.alpha * float(y) + (1.0 - self.alpha) * prev_y

        dx = self._filtered_x - prev_x
        dy = self._filtered_y - prev_y
        if (dx * dx + dy * dy) ** 0.5 < self.min_movement:
            return self._last_heading

        heading = heading_from_dxdy(dx, dy)
        self._last_heading = heading
        return heading
