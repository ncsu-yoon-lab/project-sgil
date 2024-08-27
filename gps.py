class GPSDataPoint:
    """
    A data point from the GPS containing latitude, longitude, heading (in degrees, CCW positive from East.) and
    timestamp (in milliseconds).
    TODO: verify units here
    """

    def __init__(self, latitude: float, longitude: float, heading: float, timestamp: float):
        self.latitude = latitude
        self.longitude = longitude
        self.heading = heading
        self.timestamp = timestamp

class GPS:

    def __init__(self):
        # Set up gps
        pass

    def get_location(self) -> GPSDataPoint:
        """
        Get the current location reported by the GPS.
        :return: A tuple of (latitude, longitude, heading)
        """
        return GPSDataPoint(35.77127383, -78.67416417, 116.1800003, 0)
