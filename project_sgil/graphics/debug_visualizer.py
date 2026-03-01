"""Tree visualization module for debugging area of interest and tree matching
algorithms.

file: tree_visualizer.py
author: Cole Malinchock and Jack Elia
"""

import math
import os

import matplotlib

from project_sgil.constants import AOI_ANGLE_DEG, AOI_RADIUS_M, ORIGIN, SATELLITE_IMAGE_PATH, PLOT_RANGE
from project_sgil.data_structs import Point, Pose2d, Tree, Wedge
from project_sgil.utils.converter import Converter
from project_sgil.utils.utils import get_relative_angle

matplotlib.use("Agg")  # Use non-interactive backend that won't interfere with OpenCV
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class DebugVisualizer:
    """Visualization utilities for tree matching and area of interest
    debugging."""

    # Class-level output directory and counter so static methods can save files.
    OUTPUT_DIR: str = "debug_plots"
    _PLOT_COUNTER: int = 0

    @staticmethod
    def draw_sat_background(
        ax: plt.Axes,
        img: np.ndarray,
        cx: float,
        cy: float,
        view_w_px: int,
        view_h_px: int,
        zoom: float = 1.0,
        alpha: float = 0.18,
    ) -> None:

        # Build a throw-away converter just to read the image's world bounds.
        converter = Converter(ORIGIN[0], ORIGIN[1])

        # extent = [left, right, bottom, top] in axes (world) coordinates.
        # y_min is the southernmost edge, y_max the northernmost — matching
        # the conventional image orientation where row 0 is the top of the
        # geographic area.
        extent = [converter.x_min, converter.x_max, converter.y_min, converter.y_max]

        ax.imshow(
            img,
            extent=extent,
            origin="upper",   # row 0 of the array = top of the geographic area
            aspect="equal",
            alpha=alpha,
            interpolation="bilinear",
            zorder=0,         # keep the image behind every other artist
        )

        # Set the visible window centred on (cx, cy).
        # zoom=1 → the window is exactly view_w_px × view_h_px satellite pixels wide/tall.
        half_w = (view_w_px / 2.0) / zoom
        half_h = (view_h_px / 2.0) / zoom
        ax.set_xlim(cx - 20, cx + 20)
        ax.set_ylim(cy - 20, cy + 30)

    @staticmethod
    def _ensure_output_dir() -> None:
        """Create output directory for plots if it doesn't exist."""
        if not os.path.exists(DebugVisualizer.OUTPUT_DIR):
            os.makedirs(DebugVisualizer.OUTPUT_DIR)

    @classmethod
    def _save_figure(cls, fig: plt.Figure, filename: str) -> str:
        """Save figure to the output directory with bookkeeping."""
        
        cls._ensure_output_dir()
        filepath = os.path.join(cls.OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        return filepath

    @classmethod
    def _next_name(cls, stem: str, suffix: str) -> str:
        """Generate a sequential filename like '{stem}_{NNN}{suffix}'."""
        name = f"{stem}_{cls._PLOT_COUNTER:03d}{suffix}"
        cls._PLOT_COUNTER += 1
        return name

    @staticmethod
    def plot_aoi(
        all_sat_tree_loc: list[Point],
        aoi_sat_trees: list[Point],
        current_pose: Pose2d,
        save_name: str | None = None,
    ) -> None:
        """Visualize area of interest trees for debugging purposes.

        :param all_sat_tree_loc: List of all satellite tree locations as
            Point.
        :param aoi_sat_trees: List of trees within the area of interest
            as Point.
        :param current_pose: Current position as Pose2d object.
        :param save_name: Optional custom name for saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        converter = Converter(ORIGIN[0], ORIGIN[1])
        pose_lat, pose_lon = converter.xy_to_latlon(Point(current_pose.x, current_pose.y))

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
        ax.text(
            current_pose.x + 0.5,
            current_pose.y + 0.5,
            f"({pose_lat:.6f}, {pose_lon:.6f})",
            fontsize=7,
            alpha=0.8,
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

        # Zoom in around the AOI with a small margin
        margin = AOI_RADIUS_M * 1.2
        ax.set_xlim(current_pose.x - margin, current_pose.x + margin)
        ax.set_ylim(current_pose.y - margin, current_pose.y + margin)

        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Area of Interest Debug Plot")
        ax.legend()

        stats_text = (
            f"Total trees: {len(all_sat_tree_loc)}\n"
            f"AOI trees: {len(aoi_sat_trees)}\n"
            f"Position: ({current_pose.x:.1f}, {current_pose.y:.1f})\n"
            f"Heading: {current_pose.yaw:.1f}°\n"
            f"LatLon: ({pose_lat:.6f}, {pose_lon:.6f})"
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

        # Save to disk
        filename = (
            f"{save_name}_aoi.png" if save_name else DebugVisualizer._next_name("aoi_plot", ".png")
        )
        path = DebugVisualizer._save_figure(fig, filename)
        print(f"AOI plot saved to: {path}")

    @staticmethod
    def plot_wedges(
        wedges: list[Wedge],
        wedge_combination: dict[Wedge, Tree],
        current_pose: Pose2d,
        rtk_pose: Pose2d,
        aoi_trees: list[Tree],
        estimated_location: Point,
        save_name: str | None = None,
    ) -> None:
        """Visualize AOI trees, wedge directions, and chosen tree per wedge."""
        
        fig, ax = plt.subplots(figsize=(10, 8))

        if SATELLITE_IMAGE_PATH is not None:
            
            img = mpimg.imread(SATELLITE_IMAGE_PATH)

            VIEW_W_PX = 1600
            VIEW_H_PX = 1200
            DPI = 200
            fig = plt.figure(figsize=(VIEW_W_PX / DPI, VIEW_H_PX / DPI), dpi=DPI)
            ax = fig.add_subplot(111)

            # Example: put current_pose centered in x, and 15 m "up" in world y
            cx = current_pose.x
            cy = current_pose.y + 15.0
            
            DebugVisualizer.draw_sat_background(
                ax,
                img,
                cx=cx,
                cy=cy,
                view_w_px=VIEW_W_PX,
                view_h_px=VIEW_H_PX,
                zoom=1.0,       # 1.0 => ~1 plot pixel = 1 sat pixel
                alpha=0.18,
            )

        converter = Converter(ORIGIN[0], ORIGIN[1])
        
        pose_lat, pose_lon = converter.xy_to_latlon(Point(current_pose.x, current_pose.y))
        labeled_ids: set[int] = set()

        def _label_tree(tree: Tree) -> None:
            tree_id = getattr(tree, "id", None)
            if tree_id is not None and tree_id in labeled_ids:
                return
            lat, lon = converter.xy_to_latlon(Point(tree.x, tree.y))
            ax.text(
                tree.x + 0.5,
                tree.y + 0.5,
                f"({lat:.6f}, {lon:.6f})",
                fontsize=6,
                alpha=0.7,
            )
            if tree_id is not None:
                labeled_ids.add(tree_id)

        # --- AOI: faint background points for context ---
        if aoi_trees:
            ax.scatter(
                [t.x for t in aoi_trees],
                [t.y for t in aoi_trees],
                s=18,
                alpha=0.25,
                c="#777777",
                label="AOI Trees",
            )
            for t in aoi_trees:
                _label_tree(t)

        # --- Pose + heading arrow ---
        ax.scatter(
            current_pose.x, current_pose.y, s=90, marker="*", label="Current Pose", c="#1f77b4"
        )
        ax.text(
            current_pose.x + 0.5,
            current_pose.y + 0.5,
            f"({pose_lat:.6f}, {pose_lon:.6f})",
            fontsize=7,
            alpha=0.8,
        )
        heading_rad = math.radians(current_pose.yaw)
        head_len = 6.0
        ax.plot(
            [current_pose.x, current_pose.x + head_len * math.cos(heading_rad)],
            [current_pose.y, current_pose.y + head_len * math.sin(heading_rad)],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            c="#1f77b4",
        )
        
        if rtk_pose is not None:
            ax.scatter(
                rtk_pose.x,
                rtk_pose.y,
                s=70,
                marker="o",
                label="RTK Pose",
                c="#ff7f0e",
                zorder=5,
            )
            # optional: tiny label so you can see the numeric point quickly
            ax.text(
                rtk_pose.x + 0.5,
                rtk_pose.y + 0.5,
                f"RTK ({rtk_pose.x:.1f}, {rtk_pose.y:.1f})",
                fontsize=7,
                alpha=0.8,
            )

        # --- Color palette per-wedge (cycles) ---
        colors = [
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # --- For each wedge: draw guideline, candidates, and selected tree ---
        for idx, wedge in enumerate(wedges):
            color = colors[idx % len(colors)]

            # (1) Guideline (dashed) for wedge direction from the current pose
            dir_from_pose = heading_rad - math.radians(-wedge.theta_degrees)
            guide_len = 22.0
            ax.plot(
                [current_pose.x, current_pose.x + guide_len * math.cos(dir_from_pose)],
                [current_pose.y, current_pose.y + guide_len * math.sin(dir_from_pose)],
                linestyle="--",
                linewidth=1.8,
                alpha=0.8,
                c=color,
                label=(f"Wedge {idx + 1} Δθ={wedge.theta_degrees:+.1f}°" if idx == 0 else None),
            )

            # (2) All candidate trees for this wedge (small dots in wedge color)
            if wedge.trees:
                ax.scatter(
                    [t.x for t in wedge.trees],
                    [t.y for t in wedge.trees],
                    s=28,
                    alpha=0.6,
                    c=color,
                )
                for t in wedge.trees:
                    _label_tree(t)

            # (3) If this wedge is in the selected combination, highlight that pick
            if wedge in wedge_combination:
                sel = wedge_combination[wedge]
                ax.scatter(
                    sel.x,
                    sel.y,
                    s=80,
                    marker="s",
                    edgecolor="k",
                    linewidths=1.0,
                    alpha=0.95,
                    c=color,
                    label=(f"Pick for Wedge {idx + 1}" if idx == 0 else None),
                )

                # Draw a ray through the selected tree along the relative line-of-bearing
                theta_rel = math.radians(-wedge.theta_degrees)
                ray_dir = (heading_rad - theta_rel - math.radians(180.0)) % (2 * math.pi)
                ray_len = 28.0
                ax.plot(
                    [sel.x, sel.x + ray_len * math.cos(ray_dir)],
                    [sel.y, sel.y + ray_len * math.sin(ray_dir)],
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                    c=color,
                )

        # --- Estimated location (solution) ---
        ax.scatter(
            estimated_location.x,
            estimated_location.y,
            s=110,
            marker="*",
            label="Estimated Position",
            c="#d62728",
        )

        # --- Cosmetics ---
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Wedge Selection & Geometry")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = (
            f"{save_name}.png" if save_name else DebugVisualizer._next_name("wedges_plot", ".png")
        )
        
        path = DebugVisualizer._save_figure(fig, filename)
        
        print(f"Wedges plot saved to: {path}")

    @staticmethod
    def plot_thetas(
        ground_thetas: list[float],
        current_pose: Pose2d,
        aoi_trees: list[Tree],
        save_name: str | None = None,
    ) -> None:
        """Plot pose, AOI trees, and theta lines for ground-view angles.

        :param ground_thetas: Relative angles to trees (positive left, negative right).
        :param current_pose: RTK pose used as the ray origin.
        :param aoi_trees: AOI background trees for context.
        :param save_name: Optional custom filename stem.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        converter = Converter(ORIGIN[0], ORIGIN[1])
        pose_lat, pose_lon = converter.xy_to_latlon(Point(current_pose.x, current_pose.y))
        labeled_ids: set[int] = set()

        def _label_tree(tree: Tree) -> None:
            tree_id = getattr(tree, "id", None)
            if tree_id is not None and tree_id in labeled_ids:
                return
            lat, lon = converter.xy_to_latlon(Point(tree.x, tree.y))
            ax.text(
                tree.x + 0.5,
                tree.y + 0.5,
                f"({lat:.6f}, {lon:.6f})",
                fontsize=6,
                alpha=0.7,
            )
            if tree_id is not None:
                labeled_ids.add(tree_id)

        # AOI background trees
        if aoi_trees:
            ax.scatter(
                [t.x for t in aoi_trees],
                [t.y for t in aoi_trees],
                s=18,
                alpha=0.25,
                c="#777777",
                label="AOI Trees",
            )
            for t in aoi_trees:
                _label_tree(t)

        # Pose + heading
        ax.scatter(
            current_pose.x,
            current_pose.y,
            s=90,
            marker="*",
            label="Current Pose",
            c="#1f77b4",
        )
        ax.text(
            current_pose.x + 0.5,
            current_pose.y + 0.5,
            f"({pose_lat:.6f}, {pose_lon:.6f})",
            fontsize=7,
            alpha=0.8,
        )
        heading_rad = math.radians(current_pose.yaw)
        head_len = 6.0
        ax.plot(
            [current_pose.x, current_pose.x + head_len * math.cos(heading_rad)],
            [current_pose.y, current_pose.y + head_len * math.sin(heading_rad)],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            c="#1f77b4",
            label="RTK Heading",
        )

        # Theta rays (relative to heading)
        if ground_thetas:
            ray_len = 28.0
            for idx, theta in enumerate(ground_thetas):
                dir_from_pose = heading_rad - math.radians(-theta)
                end_x = current_pose.x + ray_len * math.cos(dir_from_pose)
                end_y = current_pose.y + ray_len * math.sin(dir_from_pose)
                ax.plot(
                    [current_pose.x, end_x],
                    [current_pose.y, end_y],
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.8,
                    c="#2ca02c",
                    label=("Ground Thetas" if idx == 0 else None),
                )
                mid_x = current_pose.x + 0.5 * ray_len * math.cos(dir_from_pose)
                mid_y = current_pose.y + 0.5 * ray_len * math.sin(dir_from_pose)
                ax.text(
                    mid_x + 0.4 * math.cos(dir_from_pose),
                    mid_y + 0.4 * math.sin(dir_from_pose),
                    f"θ={theta:+.1f}°",
                    fontsize=7,
                    alpha=0.85,
                )

        # Zoom to content with a small margin
        all_x = [current_pose.x] + [t.x for t in aoi_trees]
        all_y = [current_pose.y] + [t.y for t in aoi_trees]
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            margin = max(5.0, 0.1 * max(max_x - min_x, max_y - min_y))
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Theta Lines (Pre-Match)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = (
            f"{save_name}.png" if save_name else DebugVisualizer._next_name("theta_plot", ".png")
        )
        path = DebugVisualizer._save_figure(fig, filename)
        print(f"Theta plot saved to: {path}")

    @staticmethod
    def clear_plots(output_dir: str = "debug_plots") -> None:
        """Delete all saved plot images in the specified directory.

        :param output_dir: Directory containing plot images (default =
            "debug_plots").
        """
        if not os.path.exists(output_dir):
            print(f"No directory named '{output_dir}' found.")
            return

        removed = 0
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(filepath)
                removed += 1

        print(f"Cleared {removed} plot image(s) from '{output_dir}'.")
