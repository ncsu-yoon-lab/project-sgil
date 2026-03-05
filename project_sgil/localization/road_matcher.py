"""Road matcher / snapper.

This module is intentionally self-contained.

Right now it supports matching to a single *straight* road segment provided as
(lat, lon) endpoints. Later we can swap the road source (e.g., OSM polylines)
without changing the rest of the localization pipeline.

Contract
========
Inputs:
- raw GPS position (lat, lon) and a "gps heading" (yaw in degrees, ENU: 0°=East,
  CCW positive)
Outputs:
- snapped pose in local XY, plus a road-aligned yaw chosen to be closest to the
  provided gps heading (either up-road or down-road).

author: Jack Elia (project) / Copilot (implementation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from project_sgil.data_structs import Pose2d
from project_sgil.utils.converter import Converter


@dataclass(frozen=True)
class RoadMatch:
    """Result of snapping a point to the road."""

    snapped_pose: Pose2d
    snap_distance_m: float
    road_yaw_deg: float


class RoadMatcher:
    """Snap GPS positions to a (currently) straight road segment and derive heading."""

    # TODO: replace with a configurable road source.
    # These placeholders are intentionally obvious so they can't be mistaken for
    # real data.
    ROAD_START_LATLON: tuple[float, float] = (35.772922, -78.639484)
    ROAD_END_LATLON: tuple[float, float] = (35.775636, -78.639340)

    def __init__(
        self,
        converter: Converter,
        road_start_latlon: tuple[float, float] | None = None,
        road_end_latlon: tuple[float, float] | None = None,
    ) -> None:
        self._converter = converter
        self._road_start_latlon = road_start_latlon or self.ROAD_START_LATLON
        self._road_end_latlon = road_end_latlon or self.ROAD_END_LATLON

        self._ax, self._ay = self._converter.latlon_to_xy(self._road_start_latlon)
        self._bx, self._by = self._converter.latlon_to_xy(self._road_end_latlon)

        dx = self._bx - self._ax
        dy = self._by - self._ay
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            raise ValueError(
                "RoadMatcher road endpoints are identical (degenerate segment). "
                "Set ROAD_START_LATLON/ROAD_END_LATLON or pass endpoints to the constructor."
            )

        # "Forward" yaw along A->B in ENU degrees (0°=East, CCW positive)
        self._yaw_ab = math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _wrap_deg(angle: float) -> float:
        """Wrap angle to (-180, 180]."""
        a = (float(angle) + 180.0) % 360.0 - 180.0
        # keep +180 instead of -180 for nicer diffs
        return 180.0 if a == -180.0 else a

    @classmethod
    def _ang_diff_deg(cls, a: float, b: float) -> float:
        """Smallest signed difference a-b in degrees in (-180, 180]."""
        return cls._wrap_deg(float(a) - float(b))

    def _closest_point_on_segment(
        self, px: float, py: float
    ) -> tuple[float, float, float]:
        """Return (cx, cy, t) where C is closest point on segment AB to P.

        t is the clamped projection parameter in [0, 1].
        """
        ax, ay, bx, by = self._ax, self._ay, self._bx, self._by
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby

        # denom can't be 0 (checked in __init__), but keep it safe
        t = 0.0 if denom <= 0.0 else (apx * abx + apy * aby) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        cx = ax + t * abx
        cy = ay + t * aby
        return cx, cy, t

    def match_gps(
        self,
        gps_latlon: tuple[float, float],
        gps_yaw_deg: float,
    ) -> RoadMatch:
        """Snap the GPS position to the road and pick a road yaw closest to gps yaw."""
        px, py = self._converter.latlon_to_xy(gps_latlon)
        cx, cy, _t = self._closest_point_on_segment(px, py)
        snap_dist = float(math.hypot(px - cx, py - cy))

        # Two candidate headings: along A->B and B->A
        yaw1 = float(self._yaw_ab)
        yaw2 = self._wrap_deg(yaw1 + 180.0)

        diff1 = abs(self._ang_diff_deg(gps_yaw_deg, yaw1))
        diff2 = abs(self._ang_diff_deg(gps_yaw_deg, yaw2))
        chosen_yaw = yaw1 if diff1 <= diff2 else yaw2

        snapped_pose = Pose2d(x=float(cx), y=float(cy), yaw=float(chosen_yaw))
        return RoadMatch(
            snapped_pose=snapped_pose,
            snap_distance_m=snap_dist,
            road_yaw_deg=float(chosen_yaw),
        )
