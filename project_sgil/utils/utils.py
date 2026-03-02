import math
import statistics
from dataclasses import dataclass, field


@dataclass
class Point:
    """Represents a point in 2D space.

    :param x: The x-coordinate.
    :param y: The y-coordinate.
    """

    x: float
    y: float


@dataclass
class Tree(Point):
    """Represents a tree in 2D space.

    :param x: The x-coordinate of the tree.
    :param y: The y-coordinate of the tree.
    :param id: A unique identifier for the tree.
    """

    id: int


@dataclass(eq=False)
class Wedge:
    """Represents an angular wedge containing multiple trees and an optional
    matched tree.

    :param theta_degrees: The angle of the wedge in degrees.
    :param trees: A list of Tree instances contained within the wedge.
    :param matched_tree: An optional Tree that has been matched within
        the wedge; defaults to None.
    """

    theta_degrees: float
    trees: list[Tree] = field(default_factory=list)
    matched_tree: Tree | None = None


@dataclass
class Pose2d(Point):
    """Represents a 2D pose with position and yaw.

    :param x: The x-coordinate of the pose.
    :param y: The y-coordinate of the pose.
    :param yaw: The yaw angle (in degrees) of the pose.
    """

    yaw: float

    def __post_init__(self) -> None:
        # Normalize yaw to be within [-180, 180) degrees
        self.yaw = (self.yaw + 180.0) % 360.0 - 180.0
        if self.yaw == -180.0:
            self.yaw = 180.0


@dataclass
class PoseEstimate:
    """Represents a pose estimate with a Pose2d and score.

    :param pose: The Pose2d representing the estimated position and
        orientation.
    :param score: The score calculated heuristically for the pose
        estimate. Higher is better. It represents how likely that this
        pose estimate was made with a correct matching of trees.
    :param confidence: A float between 0.0 and 1.0 representing the
        confidence in the pose estimate. This is independent of the
        score and indicative of the reliability of the estimate.
    :param combo_index: The 1-based index of the wedge/tree combination that
        produced this estimate (matches the 'combo_{N}_...' used in debug plot names).
    """

    pose: Pose2d
    score: float
    confidence: float
    wedge_combinations: dict[Wedge, Tree]
    combo_index: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class LocalizationResult:
    """Container for per-image localization results (for final table).

    :param image_name: Image filename processed.
    :param sgil_err_m: Distance (m) between SGIL estimate and RTK ground
        truth.
    :param rtk_pose: RTK-derived pose (XY in local frame, yaw degrees).
    :param current_pose: Pose passed into the tree matcher for this
        image.
    :param estimated_pose: Pose using estimated XY (from tree matcher)
        and current yaw. Can be None when no estimate was produced.
    """

    image_name: str
    sgil_err_m: float
    gps_err_m: float
    rtk_pose: Pose2d
    gps_pose: Pose2d | None
    current_pose: Pose2d
    estimated_pose: Pose2d | None
    matched: bool


def normalize_deg(angle: float) -> float:
    """Normalize an angle to [-180, 180) degrees."""
    normalized = (angle + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def heading_from_dxdy(dx: float, dy: float) -> float:
    """Return heading in degrees for a delta vector (0=N, 90=E)."""
    heading_rad = math.atan2(dx, dy)
    return normalize_deg(math.degrees(heading_rad))


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
    return normalize_deg(rel_angle_deg)


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
