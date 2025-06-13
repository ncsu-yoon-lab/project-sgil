"""
This script handles the conversion between the lat, lon to the x, y

file: converter.py
author: Cole Malinchock
"""

# Importing necessary libraries
import math

# Custom libraries
from constants import *


class Converter:
    """
    The Converter class that handles the conversion between given factors
    """
    
    def __init__(self, lat_origin: float, lon_origin: float) -> None:
        """
        Initialization method for setting the home lat, lon

        Args:
            lat_origin: The origin latitude
            lon_origin: The origin longitude
        """

        # Sets the origin of the coordinate system
        self.origin = (lat_origin, lon_origin)


    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        The haversine function to handle calculating distance between two lat, lon points

        Args:
            lat1: the first latitude point
            lon1: the first longitude point
            lat2: the second latitude point
            lon2: the second longitude point

        Returns:
            distance: the distance between the two points in meters
        """

        # Finds the difference between the lat longs and converts them to radians
        d_lat = math.radians(lat1 - lat2)
        d_lon = math.radians(lon1 - lon2)

        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat2)) * math.cos(math.radians(lat1)) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = EARTH_RADIUS_M * c

        return distance


    def latlon_to_xy(self, point: tuple[float, float]) -> tuple[float, float]:
        """
        Converts the lat, lon to the x, y grid

        Args:
            point: as the lat, lon point
        
        Returns:
            the x, y point of the input lat, lon
        """
        
        # Gets the lat, lon point
        lat = float(point[0])
        lon = float(point[1])

        # Uses haversine to get the x, y points
        x = self.haversine(ORIGIN[0], lon, self.origin[0], self.origin[1])
        y = self.haversine(lat, self.origin[1], self.origin[0], self.origin[1])

        # Changes the x and y to negative if they are below/left of the origin
        if lon < self.origin[1]:
            x = -x
        if lat < self.origin[0]:
            y = -y

        return (x, y)


    def xy_to_latlon(self, point: tuple[float, float]) -> tuple[float, float]:
        """
        Convert x and y back to latitude and longitude

        Args:
            point: the x, y point to be converted

        Returns:
            the lat, lon point converted from the x, y
        """

        # The x and y points
        x = point[0]
        y = point[1]

        # Gets the change in lat, lon from the given x, y relative to the Earth radius
        delta_lat = y / EARTH_RADIUS_M
        delta_lon = x / (EARTH_RADIUS_M * math.cos(math.radians(self.origin[0])))

        # Gets the lat, lon from the origin
        lat = self.origin[0] + math.degrees(delta_lat)
        lon = self.origin[1] + math.degrees(delta_lon)

        return (lat, lon)


    def image_x_to_theta(self, x: float, image_width: int=1280) -> float:
        """
        Given the x position, it outputs the theta to that x position in the image

        Args:
            x: the x position in the image
            image_width: the width of the image in pixels

        Returns:
            theta: the ground theta from the camera view to the selected x
        """

        # Calculate center of image
        center_x = image_width / 2

        # Calculate pixel offset from center
        offset_x = x - center_x

        # Calculate half the FOV
        dpp = H_FOV_DEG / 2

        # Calculate theta
        theta = dpp * offset_x / center_x * -1.0

        return theta

    def heading_to_yaw(self, heading: float) -> float:
        """
        Gets the yaw from the heading

        Args:
            heading: the heading read from the gps
        
        Returns:
            yaw: the converted heading to the x, y coordinate grid in degrees
        """

        # Subtract the offset from the heading
        yaw = (450 - heading) % 360

        return yaw
