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

    # Optional second road segment. Set these to enable snapping across two roads.
    # If you don't want a second road, leave them as None.
    ROAD2_START_LATLON: tuple[float, float] = (35.775595, -78.638384)
    ROAD2_END_LATLON: tuple[float, float] = (35.775773, -78.643686

)

    def __init__(
        self,
        converter: Converter,
        road_start_latlon: tuple[float, float] | None = None,
        road_end_latlon: tuple[float, float] | None = None,
        *,
        road2_start_latlon: tuple[float, float] | None = None,
        road2_end_latlon: tuple[float, float] | None = None,
    ) -> None:
        self._converter = converter

        # Road 1 endpoints
        self._road_start_latlon = road_start_latlon or self.ROAD_START_LATLON
        self._road_end_latlon = road_end_latlon or self.ROAD_END_LATLON

        # Road 2 endpoints (optional)
        self._road2_start_latlon = road2_start_latlon or self.ROAD2_START_LATLON
        self._road2_end_latlon = road2_end_latlon or self.ROAD2_END_LATLON

        # Convert road 1 to XY
        self._ax, self._ay = self._converter.latlon_to_xy(self._road_start_latlon)
        self._bx, self._by = self._converter.latlon_to_xy(self._road_end_latlon)
        dx1 = self._bx - self._ax
        dy1 = self._by - self._ay
        if abs(dx1) < 1e-9 and abs(dy1) < 1e-9:
            raise ValueError(
                "RoadMatcher road endpoints are identical (degenerate segment). "
                "Set ROAD_START_LATLON/ROAD_END_LATLON or pass endpoints to the constructor."
            )
        self._yaw_ab = math.degrees(math.atan2(dy1, dx1))

        # Convert road 2 to XY if provided
        self._has_road2 = self._road2_start_latlon is not None and self._road2_end_latlon is not None
        if self._has_road2:
            self._a2x, self._a2y = self._converter.latlon_to_xy(self._road2_start_latlon)
            self._b2x, self._b2y = self._converter.latlon_to_xy(self._road2_end_latlon)
            dx2 = self._b2x - self._a2x
            dy2 = self._b2y - self._a2y
            if abs(dx2) < 1e-9 and abs(dy2) < 1e-9:
                raise ValueError(
                    "RoadMatcher road2 endpoints are identical (degenerate segment). "
                    "Set ROAD2_START_LATLON/ROAD2_END_LATLON or pass road2 endpoints to the constructor."
                )
            self._yaw2_ab = math.degrees(math.atan2(dy2, dx2))
        else:
            self._a2x = self._a2y = self._b2x = self._b2y = 0.0
            self._yaw2_ab = 0.0

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
        """Snap the GPS position to the closest configured road and pick road yaw."""
        px, py = self._converter.latlon_to_xy(gps_latlon)

        # Snap to road 1
        cx1, cy1, _t1 = self._closest_point_on_segment(px, py)
        d1 = float(math.hypot(px - cx1, py - cy1))

        best_cx, best_cy, best_yaw_ab, best_dist = cx1, cy1, float(self._yaw_ab), d1

        # Optionally snap to road 2 and pick whichever is closer
        if self._has_road2:
            # Temporarily compute closest point to road2 segment using local helper math
            ax, ay, bx, by = self._a2x, self._a2y, self._b2x, self._b2y
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            denom = abx * abx + aby * aby
            t = 0.0 if denom <= 0.0 else (apx * abx + apy * aby) / denom
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cx2 = ax + t * abx
            cy2 = ay + t * aby
            d2 = float(math.hypot(px - cx2, py - cy2))

            if d2 < best_dist:
                best_cx, best_cy, best_yaw_ab, best_dist = float(cx2), float(cy2), float(self._yaw2_ab), d2

        # Two candidate headings: along A->B and B->A for the chosen road
        yaw1 = float(best_yaw_ab)
        yaw2 = self._wrap_deg(yaw1 + 180.0)

        diff1 = abs(self._ang_diff_deg(gps_yaw_deg, yaw1))
        diff2 = abs(self._ang_diff_deg(gps_yaw_deg, yaw2))
        chosen_yaw = yaw1 if diff1 <= diff2 else yaw2

        snapped_pose = Pose2d(x=float(best_cx), y=float(best_cy), yaw=float(chosen_yaw))
        return RoadMatch(
            snapped_pose=snapped_pose,
            snap_distance_m=float(best_dist),
            road_yaw_deg=float(chosen_yaw),
        )
