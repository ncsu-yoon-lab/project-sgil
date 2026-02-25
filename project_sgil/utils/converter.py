import math

from project_sgil.constants import EARTH_RADIUS_M, H_FOV_DEG
from project_sgil.data_structs import Point
from project_sgil.utils.utils import normalize_deg

# TODO: make this class use our data classes, and organize methods by private, static, and public


class Converter:
    """Converter for geographic and image coordinates.

    Provides methods for converting between latitude/longitude and
    Cartesian coordinates, as well as image x-coordinate to angle and
    compass heading to yaw.
    """

    def __init__(self, lat_origin: float, lon_origin: float) -> None:
        """Initialize the Converter with a geographic origin.

        :param lat_origin: Latitude of the origin in degrees.
        :param lon_origin: Longitude of the origin in degrees.
        """
        self.origin: tuple[float, float] = (lat_origin, lon_origin)

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute the great-circle distance between two points on Earth.

        Uses the haversine formula.

        :param lat1: Latitude of first point in degrees.
        :param lon1: Longitude of first point in degrees.
        :param lat2: Latitude of second point in degrees.
        :param lon2: Longitude of second point in degrees.
        :return: Distance between points in meters.
        """
        d_lat: float = math.radians(lat1 - lat2)
        d_lon: float = math.radians(lon1 - lon2)

        a: float = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat2)) * math.cos(math.radians(lat1)) * math.sin(d_lon / 2) ** 2
        )
        c: float = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return EARTH_RADIUS_M * c

    def latlon_to_xy(self, point: tuple[float, float]) -> tuple[float, float]:
        """Convert latitude/longitude to local Cartesian x, y coordinates.

        :param point: Tuple of (latitude, longitude) in degrees.
        :return: Tuple of (x, y) in meters relative to the origin.
        """
        lat = math.radians(float(point[0]))
        lon = math.radians(float(point[1]))
        origin_lat = math.radians(self.origin[0])
        origin_lon = math.radians(self.origin[1])

        # Calculate differences in radians
        d_lat = lat - origin_lat
        d_lon = lon - origin_lon

        # Convert to meters using local approximation
        y = d_lat * EARTH_RADIUS_M
        x = d_lon * EARTH_RADIUS_M * math.cos(origin_lat)

        return (x, y)

    def xy_to_latlon(self, point: Point) -> tuple[float, float]:
        """Convert local Cartesian x, y coordinates back to latitude/longitude.

        :param point: Point with x, y in meters relative to the origin.
        :return: Tuple of (latitude, longitude) in degrees.
        """
        x: float = point.x
        y: float = point.y

        delta_lat: float = y / EARTH_RADIUS_M
        delta_lon: float = x / (EARTH_RADIUS_M * math.cos(math.radians(self.origin[0])))

        lat: float = self.origin[0] + math.degrees(delta_lat)
        lon: float = self.origin[1] + math.degrees(delta_lon)

        return (lat, lon)

    @staticmethod
    def image_x_to_theta(x: float, image_width: int = 1280) -> float:
        """Map an image pixel x-coordinate to a viewing angle theta.

        Converts horizontal pixel offset to an angle using half the
        horizontal field of view (H_FOV_DEG).

        :param x: Pixel x-coordinate in image.
        :param image_width: Width of the image in pixels.
        :return: Angle theta in degrees (positive left, negative right).
        """
        center_x: float = image_width / 2
        offset_x: float = x - center_x
        half_fov: float = H_FOV_DEG / 2

        return -half_fov * offset_x / center_x

    @staticmethod
    def heading_to_yaw(heading: float) -> float:
        """Convert a compass heading to a yaw angle.

        Yaw is measured by finding heading pointing the x direction
        (parallel to vector from EB1 to EB3).

        :param heading: Compass heading in degrees (0=N, 90=E).
        :return: Yaw angle in degrees where 0 is +x axis.
        """
        return normalize_deg(450 - heading)
