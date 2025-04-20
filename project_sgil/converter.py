import math

from constants import *

class Converter():

    def __init__(self, lat_origin, lon_origin):

        # Sets the origin of the coordinate system
        self.origin = (lat_origin, lon_origin)

    def haversine(self, lat1, lon1, lat2, lon2):

        # Finds the difference between the lat longs and converts them to radians
        d_lat = math.radians(lat1 - lat2)
        d_lon = math.radians(lon1 - lon2)

        a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat2)) * math.cos(math.radians(lat1)) * math.sin(d_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = EARTH_RADIUS_M * c

        return distance

    def latlon_to_xy(self, point):

        lat = float(point[0])
        lon = float(point[1])

        x = self.haversine(ORIGIN[0], lon, self.origin[0], self.origin[1])
        y = self.haversine(lat, self.origin[1], self.origin[0], self.origin[1])

        if lon < self.origin[1]:
            x = -x
        if lat < self.origin[0]:
            y = -y

        return (x, y)

    def xy_to_latlon(self, x, y):
        # Convert x and y back to latitude and longitude
        # delta_lat and delta_lon are changes in lat and lon from the origin
        delta_lat = y / EARTH_RADIUS_M
        delta_lon = x / (EARTH_RADIUS_M * math.cos(math.radians(self.origin[0])))

        lat = self.origin[0] + math.degrees(delta_lat)
        lon = self.origin[1] + math.degrees(delta_lon)

        return (lat, lon)
    
    def image_x_to_theta(self, x, image_width=1280):

        # Calculate center of image
        center_x = image_width / 2
        
        # Calculate pixel offset from center
        offset_x = x - center_x
        
        # Calculate half the FOV
        dpp = H_FOV_DEG / 2
        
        # Calculate theta
        theta = dpp * offset_x / center_x
        
        return theta

    def heading_to_yaw(self, heading) -> float:

        # Measured by finding heading pointing the x direction (parallel to vector from EB1 to EB3)
        # Degrees
        OFFSET = 130

        # Subtract the offset from the heading
        yaw = heading - OFFSET

        # If the yaw is still greater than 180, subtract 360 from it so that it can be converted from a range of (0, 360) to (-180, 180) which is required by pure pursuit
        if yaw > 180:
            yaw -= 360

        return yaw