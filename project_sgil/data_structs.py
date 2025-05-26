"""
This script handles all global data structs that will be used throughout the scripts

file: data_structs.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
from dataclasses import dataclass, field


@dataclass
class Pose2d:
    """
    Class for the 2 dimensional pose
    """

    x: float        # The x position
    y: float        # The y position
    theta: float    # The heading


@dataclass
class Wedge:
    """
    The wedge object for handling the tree matching
    """

    ground_theta_deg: float                                             # The theta from the ground view
    trees_xy: list[tuple[float, float]] = field(default_factory=list)   # The x, y position of the trees in the wedge
    matched_tree: tuple[float, float] | None = None                     # The tree that is matched to the wedge
