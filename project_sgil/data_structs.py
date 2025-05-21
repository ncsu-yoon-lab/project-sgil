from dataclasses import dataclass, field

@dataclass
class Pose2d:
    x: float
    y: float
    theta: float


@dataclass
class Wedge:
    ground_theta_deg: float
    trees_xy: list[tuple[float, float]] = field(default_factory=list)
    matched_tree: tuple[float, float] | None = None
