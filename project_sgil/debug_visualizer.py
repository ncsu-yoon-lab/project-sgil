"""
Tree visualization module for debugging area of interest and tree matching
algorithms.

file: tree_visualizer.py
author: Cole Malinchock and Jack Elia
"""

# Import necessary libraries
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that won't interfere with OpenCV
import matplotlib.pyplot as plt
import os
from constants import *
from data_structs import Pose2d


class DebugVisualizer:
    """
    Visualization utilities for tree matching and area of interest debugging.
    """

    def __init__(self) -> None:
        """
        Initialize the DebugVisualizer.
        """
        # Create output directory for plots if it doesn't exist
        self.output_dir = "debug_plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.plot_counter = 0

    def plot_aoi(self, all_sat_tree_loc: list[tuple[float, float]], 
                 aoi_sat_trees: list[tuple[float, float]], 
                 current_pose: Pose2d, 
                 save_name: str = None) -> None:
        """
        Visualize area of interest trees for debugging purposes.

        :param all_sat_tree_loc: List of all satellite tree locations.
        :param aoi_sat_trees: List of trees within the area of interest.
        :param current_pose: Current position as Pose2d object.
        :param save_name: Optional custom name for saved plot.
        """

        # Creates the figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all satellite trees in blue
        all_x = [tree[0] for tree in all_sat_tree_loc]
        all_y = [tree[1] for tree in all_sat_tree_loc]
        ax.scatter(all_x, all_y, s=20, c="blue", alpha=0.5, label="All Trees")

        # Plot AOI trees in green
        if aoi_sat_trees and len(aoi_sat_trees) > 0:
            aoi_x = [tree[0] for tree in aoi_sat_trees]
            aoi_y = [tree[1] for tree in aoi_sat_trees]
            ax.scatter(aoi_x, aoi_y, s=50, c="green", alpha=0.8, label="AOI Trees")

        # Plot current position in red
        if current_pose:
            ax.scatter(
                current_pose.x,
                current_pose.y,
                s=100,
                c="red",
                marker="*",
                label="Current Position",
            )

            # Draw line showing heading direction
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
                fc="red",
                ec="red",
            )

        # Add AOI radius circle
        if current_pose:
            circle = plt.Circle(
                (current_pose.x, current_pose.y),
                AOI_RADIUS_M,
                fill=False,
                color="red",
                linestyle="--",
                alpha=0.7,
            )
            ax.add_patch(circle)

        # Add field of view indicator
        if current_pose:
            heading_rad = math.radians(current_pose.yaw)
            half_fov = math.radians(AOI_ANGLE_DEG)
            start_angle = heading_rad - half_fov
            end_angle = heading_rad + half_fov

            wedge = plt.matplotlib.patches.Wedge(
                (current_pose.x, current_pose.y),
                AOI_RADIUS_M,
                math.degrees(start_angle),
                math.degrees(end_angle),
                fill=False,
                color="green",
                linestyle=":",
                alpha=0.7,
            )
            ax.add_patch(wedge)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add labels and legend
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Area of Interest Debug Plot")
        ax.legend()

        # Add stats
        stats_text = (
            f"Total trees: {len(all_sat_tree_loc)}\n"
            f"AOI trees: {len(aoi_sat_trees) if aoi_sat_trees else 0}\n"
            f"Position: ({current_pose.x:.1f}, {current_pose.y:.1f})\n"
            f"Heading: {current_pose.yaw:.1f}°"
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox={"facecolor": "white", "alpha": 0.8})

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot instead of showing it
        if save_name:
            filename = f"{save_name}_aoi.png"
        else:
            filename = f"aoi_plot_{self.plot_counter:03d}.png"
            self.plot_counter += 1
            
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Important: close the figure to free memory
        
        print(f"AOI plot saved to: {filepath}")

    def plot_vectors_and_intersections(self, vectors: list[list[tuple[float, float]]], 
                                     intersections: list[tuple[float, float]], 
                                     centroid: tuple[float, float], 
                                     current_pose: Pose2d) -> None:
        """
        Visualize vectors, their intersections, and calculated centroid.

        :param vectors: List of vectors as [[start_point, end_point], ...].
        :param intersections: List of intersection points.
        :param centroid: Calculated centroid point.
        :param current_pose: Current position as Pose2d object.
        """

        # Creates the figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot vectors
        for i, vector in enumerate(vectors):
            start_x, start_y = vector[0]
            end_x, end_y = vector[1]
            ax.plot([start_x, end_x], [start_y, end_y], 
                   linestyle='-', linewidth=2, alpha=0.7, 
                   label=f"Vector {i+1}")

        # Plot intersections
        if intersections:
            inter_x = [point[0] for point in intersections]
            inter_y = [point[1] for point in intersections]
            ax.scatter(inter_x, inter_y, s=50, c="orange", 
                      marker="x", label="Intersections")

        # Plot centroid
        if centroid:
            ax.scatter(centroid[0], centroid[1], s=100, c="purple", 
                      marker="D", label="Centroid")

        # Plot current position
        if current_pose:
            ax.scatter(current_pose.x, current_pose.y, s=100, c="red", 
                      marker="*", label="Current Position")

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add labels and legend
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Vector Intersection Analysis")
        ax.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_wedges(self, wedges: list, current_pose: Pose2d, 
                   aoi_trees: list[tuple[float, float]],
                   save_name: str = None) -> None:
        """
        Visualize wedges and their associated trees.

        :param wedges: List of Wedge objects.
        :param current_pose: Current position as Pose2d object.
        :param aoi_trees: List of trees within area of interest.
        :param save_name: Optional custom name for saved plot.
        """

        # Creates the figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot AOI trees
        if aoi_trees:
            aoi_x = [tree[0] for tree in aoi_trees]
            aoi_y = [tree[1] for tree in aoi_trees]
            ax.scatter(aoi_x, aoi_y, s=30, c="lightblue", 
                      alpha=0.6, label="AOI Trees")

        # Plot current position
        if current_pose:
            ax.scatter(current_pose.x, current_pose.y, s=100, c="red", 
                      marker="*", label="Current Position")

        # Plot wedges
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, wedge in enumerate(wedges):
            color = colors[i % len(colors)]
            
            # Plot trees in this wedge
            if wedge.trees:
                wedge_x = [tree.x for tree in wedge.trees]
                wedge_y = [tree.y for tree in wedge.trees]
                ax.scatter(wedge_x, wedge_y, s=60, c=color, 
                          marker='s', alpha=0.8, 
                          label=f"Wedge {i+1} ({wedge.theta_deg:.1f}°)")

            # Draw wedge direction line
            if current_pose:
                heading_rad = math.radians(current_pose.yaw)
                theta_rad = math.radians(wedge.theta_deg)
                direction_rad = heading_rad - theta_rad
                
                line_length = 20
                end_x = current_pose.x + line_length * math.cos(direction_rad)
                end_y = current_pose.y + line_length * math.sin(direction_rad)
                
                ax.plot([current_pose.x, end_x], [current_pose.y, end_y], 
                       color=color, linestyle='--', linewidth=2, alpha=0.7)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add labels and legend
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Wedge Analysis")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot instead of showing it
        if save_name:
            filename = f"{save_name}_wedges.png"
        else:
            filename = f"wedges_plot_{self.plot_counter:03d}.png"
            self.plot_counter += 1
            
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Important: close the figure to free memory
        
        print(f"Wedges plot saved to: {filepath}")

    def plot_estimation_comparison(self, current_pose: Pose2d, 
                                 estimated_position: tuple[float, float], 
                                 true_position: tuple[float, float] = None,
                                 save_name: str = None) -> None:
        """
        Compare estimated position with current and true positions.

        :param current_pose: Current position estimate as Pose2d object.
        :param estimated_position: Estimated position from tree matching.
        :param true_position: Optional true position for comparison.
        :param save_name: Optional custom name for saved plot.
        """

        # Creates the figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot current position
        ax.scatter(current_pose.x, current_pose.y, s=100, c="red", 
                  marker="*", label="Current Position")

        # Plot estimated position
        ax.scatter(estimated_position[0], estimated_position[1], s=100, 
                  c="green", marker="o", label="Estimated Position")

        # Plot true position if available
        if true_position:
            ax.scatter(true_position[0], true_position[1], s=100, 
                      c="blue", marker="^", label="True Position")
            
            # Draw error lines
            ax.plot([estimated_position[0], true_position[0]], 
                   [estimated_position[1], true_position[1]], 
                   'k--', alpha=0.5, label="Estimation Error")

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Add labels and legend
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Position Estimation Results")
        ax.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot instead of showing it
        if save_name:
            filename = f"{save_name}_comparison.png"
        else:
            filename = f"comparison_plot_{self.plot_counter:03d}.png"
            self.plot_counter += 1
            
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Important: close the figure to free memory
        
        print(f"Comparison plot saved to: {filepath}")