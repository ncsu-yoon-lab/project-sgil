"""
project_sgil.data_structs.

Provides some basic structs for SGIL
"""

from dataclasses import dataclass, field


@dataclass
class Tree:
    """
    Represents a tree in 2D space.

    :param x: The x-coordinate of the tree.
    :param y: The y-coordinate of the tree.
    :param id: A unique identifier for the tree.
    """

    x: float
    y: float
    id: int


@dataclass
class Wedge:
    """
    Represents an angular wedge containing multiple trees and an optional
    matched tree.

    :param theta_deg: The angle of the wedge in degrees.
    :param trees: A list of Tree instances contained within the wedge.
    :param matched_tree: An optional Tree that has been matched within
        the wedge; defaults to None.
    """

    theta_deg: float
    trees: list[Tree] = field(default_factory=list)
    matched_tree: Tree | None = None


@dataclass
class Pose2d:
    """
    Represents a 2D pose with position and yaw.

    :param x: The x-coordinate of the pose.
    :param y: The y-coordinate of the pose.
    :param yaw: The yaw angle (in degrees) of the pose.
    """

    x: float
    y: float
    yaw: float
