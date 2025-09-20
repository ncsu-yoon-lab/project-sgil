import math
import statistics

from project_sgil.data_structs import Point, Pose2d


def distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points.

    :param point1: First point as a Point instance.
    :param point2: Second point as a Point instance.
    :return: Distance between the points.
    """
    return math.hypot(point1.x - point2.x, point1.y - point2.y)


def get_relative_angle(point: Point, current_pose: Pose2d) -> float:
    """Calculate the relative angle from the current heading to a point.

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
    """Calculate standard deviation of distances from points to centroid.

    :param points: List of points as (x, y) tuples.
    :param centroid: Centroid point as (x, y) tuple.
    :return: Standard deviation of distances or None if insufficient
        data.
    """
    # Gets the distance of each point to the centroid
    distances = [distance(pt, centroid) for pt in points]

    # Calculates the standard deviation of all the distances
    return statistics.stdev(distances) if len(distances) > 1 else 0.0


def _segment_intersects_circle(
    start: Point,
    end: Point,
    center: Point,
    radius: float,
) -> bool:
    """Check if the segment from start→end intersects (or grazes) a circle.

    :param start: Segment start point.
    :param end: Segment end point.
    :param center: Circle center.
    :param radius: Circle radius.
    :return: True if the closest point on the segment to the circle
        center lies within the segment AND within radius distance of the
        center.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        # Degenerate segment: start == end. Treat as point-in-circle check.
        dist_sq = (start.x - center.x) ** 2 + (start.y - center.y) ** 2
        return dist_sq <= radius * radius

    # Project center onto the infinite line; get param t along the segment
    t = ((center.x - start.x) * dx + (center.y - start.y) * dy) / seg_len_sq

    # We only care about intersection with the finite segment (0 <= t <= 1)
    if t < 0.0 or t > 1.0:
        return False

    # Closest point on the segment to the circle center
    closest_x = start.x + t * dx
    closest_y = start.y + t * dy

    # Distance from closest point to center
    dist_sq = (closest_x - center.x) ** 2 + (closest_y - center.y) ** 2
    return dist_sq <= radius * radius
