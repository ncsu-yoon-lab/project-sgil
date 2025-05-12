"""
converter.py.
============

Utilities for converting between geodetic (lat/lon), local Cartesian (x/y),
and image‑space/heading angles.
"""

import math

from constants import EARTH_RADIUS_M, H_FOV_DEG


class Converter:
    """
    A bidirectional converter between geographic coordinates
    (latitude/longitude), local Cartesian coordinates (x, y), image pixel
    offsets to horizontal angles, and compass headings to yaw angles.

    :param lat_origin: Latitude of the local origin, in degrees.
    :param lon_origin: Longitude of the local origin, in degrees.
    """

    def __init__(self, lat_origin: float, lon_origin: float) -> None:
        self.origin: tuple[float, float] = (lat_origin, lon_origin)

    def haversine(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Compute the great‑circle distance between two points on the Earth
        using the haversine formula.

        :param lat1: Latitude of the first point, in degrees.
        :param lon1: Longitude of the first point, in degrees.
        :param lat2: Latitude of the second point, in degrees.
        :param lon2: Longitude of the second point, in degrees.
        :return: Distance between the two points, in meters.
        """
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_M * c

    def latlon_to_xy(self, point: tuple[float, float]) -> tuple[float, float]:
        """
        Project a (latitude, longitude) point into local Cartesian (x, y)
        coordinates relative to the configured origin. Positive X is eastward,
        positive Y is northward.

        :param point: A tuple of (latitude, longitude) in degrees.
        :return: Local Cartesian coordinates (x, y) in meters.
        """
        lat, lon = point
        origin_lat, origin_lon = self.origin

        # East‑west distance
        x = self.haversine(origin_lat, origin_lon, origin_lat, lon)
        if lon < origin_lon:
            x = -x

        # North‑south distance
        y = self.haversine(origin_lat, origin_lon, lat, origin_lon)
        if lat < origin_lat:
            y = -y

        return x, y

    def xy_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert local Cartesian (x, y) coordinates back to geographic
        (latitude, longitude) relative to the configured origin.

        :param x: Eastward distance from origin in meters.
        :param y: Northward distance from origin in meters.
        :return: A tuple of (latitude, longitude) in degrees.
        """
        origin_lat, origin_lon = self.origin

        delta_lat = y / EARTH_RADIUS_M
        delta_lon = x / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat)))

        lat = origin_lat + math.degrees(delta_lat)
        lon = origin_lon + math.degrees(delta_lon)
        return lat, lon

    def image_x_to_theta(self, x: float, image_width: int = 1280) -> float:
        """
        Map an image pixel X‑coordinate to a horizontal angle (theta)
        relative to the camera’s optical axis.

        :param x: Pixel X position (0 on left edge, image_width on
            right).
        :param image_width: Total width of the image in pixels.
        :return: Horizontal angle offset from center, in degrees.
        """
        center_x = image_width / 2
        offset_x = x - center_x
        half_fov = H_FOV_DEG / 2
        return (offset_x / center_x) * half_fov

    def heading_to_yaw(self, heading: float, offset: float = 130.0) -> float:
        """
        Convert a compass heading (0–360°) into a yaw angle (–180° to +180°)
        by subtracting a fixed offset and rewrapping into the (–180, +180)
        range.

        :param heading: Compass heading in degrees (0 = North, 90 =
            East).
        :param offset: Fixed offset to subtract (default: 130°).
        :return: Yaw angle in degrees, in the range (–180, +180).
        """
        yaw = heading - offset
        if yaw > 180:
            yaw -= 360
        elif yaw <= -180:
            yaw += 360
        return yaw
