"""This script handles all global data structs that will be used throughout the
scripts.

file: data_structs.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
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


@dataclass
class PoseEstimate:
    """Represents a pose estimate with a Pose2d and score.

    :param pose: The Pose2d representing the estimated position and orientation.
    :param score: The score calculated heuristically for the pose estimate. Higher is better.
    It represents how likely that this pose estimate was made with a correct matching of trees.
    :param confidence: A float between 0.0 and 1.0 representing the confidence in the pose estimate.
    This is independent of the score and indicative of the reliability of the estimate.
    """
    pose: Pose2d
    score: float
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
