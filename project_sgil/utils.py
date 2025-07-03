import math
import statistics

from project_sgil.data_structs import Point, Pose2d


def distance(point1: Point, point2: Point) -> float:
    """
    Calculate Euclidean distance between two points.

    :param point1: First point as a Point instance.
    :param point2: Second point as a Point instance.
    :return: Distance between the points.
    """
    return math.hypot(point1.x - point2.x, point1.y - point2.y)


def get_relative_angle(point: Point, current_pose: Pose2d) -> float:
    """
    Calculate the relative angle from the current heading to a point.

    :param point: Target point as a Point instance.
    :param current_pose: Current pose as a Pose2d instance, which
        includes x, y coordinates and current heading in degrees.
    :return: Relative angle in degrees (-180, 180).
    """
    abs_angle_rad = math.atan2(point.y - current_pose.y, point.x - current_pose.x)
    abs_angle_deg = math.degrees(abs_angle_rad)
    rel_angle_deg = abs_angle_deg - current_pose.yaw
    rel_angle_deg = ((rel_angle_deg + 180) % 360) - 180
    return rel_angle_deg


def std_deviation_of_distances(points: list[Point], centroid: Point) -> float:
    """
    Calculate standard deviation of distances from points to centroid.

    :param points: List of points as (x, y) tuples.
    :param centroid: Centroid point as (x, y) tuple.
    :return: Standard deviation of distances or None if insufficient
        data.
    """
    # Gets the distance of each point to the centroid
    distances = [distance(pt, centroid) for pt in points]

    # Calculates the standard deviation of all the distances
    return statistics.stdev(distances) if len(distances) > 1 else 0.0
