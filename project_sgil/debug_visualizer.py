"""
Tree visualization module for debugging area of interest and tree matching
algorithms.

file: tree_visualizer.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
import math

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend that won't interfere with OpenCV
import os

import matplotlib.pyplot as plt
from constants import *
from data_structs import Point, Pose2d, Wedge


class DebugVisualizer:
    """
    Visualization utilities for tree matching and area of interest
    debugging.
    """

    def __init__(self) -> None:
        """Initialize the DebugVisualizer."""
        # Create output directory for plots if it doesn't exist
        self.output_dir = "debug_plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.plot_counter = 0

    def plot_aoi(
        self,
        all_sat_tree_loc: list[Point],
        aoi_sat_trees: list[Point],
        current_pose: Pose2d,
        save_name: str | None = None,
    ) -> None:
        """
        Visualize area of interest trees for debugging purposes.

        :param all_sat_tree_loc: List of all satellite tree locations as
            Point.
        :param aoi_sat_trees: List of trees within the area of interest
            as Point.
        :param current_pose: Current position as Pose2d object.
        :param save_name: Optional custom name for saved plot.
        """

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all satellite trees in blue
        all_x = [tree.x for tree in all_sat_tree_loc]
        all_y = [tree.y for tree in all_sat_tree_loc]
        ax.scatter(all_x, all_y, s=20, alpha=0.5, label="All Trees")

        # Plot AOI trees in green
        if aoi_sat_trees:
            aoi_x = [tree.x for tree in aoi_sat_trees]
            aoi_y = [tree.y for tree in aoi_sat_trees]
            ax.scatter(aoi_x, aoi_y, s=50, alpha=0.8, label="AOI Trees")

        # Plot current position in red
        ax.scatter(
            current_pose.x,
            current_pose.y,
            s=100,
            marker="*",
            label="Current Position",
        )

        # Draw heading arrow
        heading_rad = math.radians(current_pose.yaw)
        arrow_length = 5
        dx = arrow_length * math.cos(heading_rad)
        dy = arrow_length * math.sin(heading_rad)
        ax.arrow(
            current_pose.x,
            current_pose.y,
            dx,
            dy,
            head_width=1,
            head_length=2,
        )

        # Add AOI radius circle
        circle = plt.Circle(
            (current_pose.x, current_pose.y),
            AOI_RADIUS_M,
            fill=False,
            linestyle="--",
            alpha=0.7,
        )
        ax.add_patch(circle)

        # Add field of view indicator
        half_fov = math.radians(AOI_ANGLE_DEG)
        start_angle = math.degrees(heading_rad - half_fov)
        end_angle = math.degrees(heading_rad + half_fov)
        wedge_patch = plt.matplotlib.patches.Wedge(
            (current_pose.x, current_pose.y),
            AOI_RADIUS_M,
            start_angle,
            end_angle,
            fill=False,
            linestyle=":",
            alpha=0.7,
        )
        ax.add_patch(wedge_patch)

        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Area of Interest Debug Plot")
        ax.legend()

        stats_text = (
            f"Total trees: {len(all_sat_tree_loc)}\n"
            f"AOI trees: {len(aoi_sat_trees)}\n"
            f"Position: ({current_pose.x:.1f}, {current_pose.y:.1f})\n"
            f"Heading: {current_pose.yaw:.1f}°"
        )
        plt.figtext(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8},
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{save_name}_aoi.png" if save_name else f"aoi_plot_{self.plot_counter:03d}.png"
        self.plot_counter += 1
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"AOI plot saved to: {filepath}")

    def plot_vectors_and_intersections(
        self,
        vectors: list[list[Point]],
        intersections: list[Point],
        centroid: Point,
        current_pose: Pose2d,
    ) -> None:
        """
        Visualize vectors, their intersections, and calculated centroid.

        :param vectors: List of vectors as [[start, end], ...].
        :param intersections: List of intersection Points.
        :param centroid: Calculated centroid Point.
        :param current_pose: Current position as Pose2d object.
        """

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot vectors
        for i, (start, end) in enumerate(vectors):
            ax.plot(
                [start.x, end.x],
                [start.y, end.y],
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                label=f"Vector {i + 1}",
            )

        # Plot intersections
        if intersections:
            ix = [p.x for p in intersections]
            iy = [p.y for p in intersections]
            ax.scatter(ix, iy, s=50, marker="x", label="Intersections")

        # Plot centroid
        ax.scatter(centroid.x, centroid.y, s=100, marker="D", label="Centroid")

        # Plot current position
        ax.scatter(current_pose.x, current_pose.y, s=100, marker="*", label="Current Position")

        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Vector Intersection Analysis")
        ax.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_wedges(
        self,
        wedges: list[Wedge],
        current_pose: Pose2d,
        aoi_trees: list[Point],
        estimated_location: Point,
        save_name: str | None = None,
    ) -> None:
        """
        Visualize wedges and their associated trees.

        :param wedges: List of Wedge objects.
        :param current_pose: Current position as Pose2d object.
        :param aoi_trees: List of trees within area of interest as
            Point.
        :param estimated_location: Estimated vehicle position as Point.
        :param save_name: Optional custom name for saved plot.
        """

        fig, ax = plt.subplots(figsize=(10, 8))
        if aoi_trees:
            ax.scatter(
                [t.x for t in aoi_trees],
                [t.y for t in aoi_trees],
                s=30,
                alpha=0.6,
                label="AOI Trees",
            )

        ax.scatter(current_pose.x, current_pose.y, s=100, marker="*", label="Current Position")

        colors = ["green", "orange", "purple", "brown", "pink", "gray"]

        # Heading line
        hr = math.radians(current_pose.yaw)
        hl = 5
        ax.plot(
            [current_pose.x, current_pose.x + hl * math.cos(hr)],
            [current_pose.y, current_pose.y + hl * math.sin(hr)],
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )

        for i, wedge in enumerate(wedges):
            colors[i % len(colors)]
            if wedge.matched_tree:
                tx, ty = wedge.matched_tree.x, wedge.matched_tree.y
                ax.scatter(
                    tx,
                    ty,
                    s=60,
                    marker="s",
                    alpha=0.8,
                    label=f"Wedge {i + 1} ({-wedge.theta_degrees:.1f}°)",
                )
                tr = math.radians(-wedge.theta_degrees)
                dr = (hr - tr - math.radians(180)) % (2 * math.pi)
                ll = 25
                ax.plot(
                    [tx, tx + ll * math.cos(dr)],
                    [ty, ty + ll * math.sin(dr)],
                    linestyle="-",
                    linewidth=2,
                    alpha=0.7,
                )
            # Wedge direction
            dr2 = hr - math.radians(-wedge.theta_degrees)
            ll2 = 20
            ax.plot(
                [current_pose.x, current_pose.x + ll2 * math.cos(dr2)],
                [current_pose.y, current_pose.y + ll2 * math.sin(dr2)],
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )

        # Estimated position
        ax.scatter(
            estimated_location.x,
            estimated_location.y,
            s=100,
            marker="*",
            label="Estimated Position",
        )

        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Wedge Analysis")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = (
            f"{save_name}_wedges.png" if save_name else f"wedges_plot_{self.plot_counter:03d}.png"
        )
        self.plot_counter += 1
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Wedges plot saved to: {filepath}")

    def plot_estimation_comparison(
        self,
        current_pose: Pose2d,
        estimated_position: Point,
        true_position: Point | None = None,
        save_name: str | None = None,
    ) -> None:
        """
        Compare estimated position with current and true positions.

        :param current_pose: Current position estimate as Pose2d object.
        :param estimated_position: Estimated position as Point.
        :param true_position: Optional true position as Point.
        :param save_name: Optional custom name for saved plot.
        """

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(current_pose.x, current_pose.y, s=100, marker="*", label="Current Position")
        ax.scatter(
            estimated_position.x,
            estimated_position.y,
            s=100,
            marker="o",
            label="Estimated Position",
        )

        if true_position:
            ax.scatter(true_position.x, true_position.y, s=100, marker="^", label="True Position")
            ax.plot(
                [estimated_position.x, true_position.x],
                [estimated_position.y, true_position.y],
                "k--",
                alpha=0.5,
                label="Estimation Error",
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Position Estimation Results")
        ax.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = (
            f"{save_name}_comparison.png"
            if save_name
            else f"comparison_plot_{self.plot_counter:03d}.png"
        )
        self.plot_counter += 1
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Comparison plot saved to: {filepath}")
