"""Manual tree + optional path annotation script.

For each image:
  1) (Optional) Select the path by clicking 4 corners in order BL, TL, BR, TR.
     Press Enter to finish; press Enter immediately if no path is visible to skip.
  2) Select all tree points (left-click), press Enter when done, or ESC to skip.

Saves CSV rows as:
  - image_filename
  - num_trees
  - tree_points        -> list of (x, y) pairs (ints)
  - path_points        -> list of 4 (x, y) pairs in order BL, TL, BR, TR, or "NO PATH"

author: Jack Elia
"""

import csv
import os
import sys

import cv2

from project_sgil.constants import IMAGE_FOLDER_PATH, OUTPUT_CSV
from project_sgil.data_structs import Point

# ----------------------- OpenCV helpers -----------------------


def init_window(window_name: str, x: int = 250, y: int = 250) -> None:
    """Create and position a persistent OpenCV window."""
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, x, y)
    cv2.waitKey(1)  # let the WM apply position


def set_click_callback(
    window_name: str,
    image_display,
    points: list[Point],
    label_prefix: str = "",
) -> None:
    """Attach a mouse callback that appends clicked points and draws
    markers."""

    def _callback(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = Point(float(x), float(y))
            points.append(pt)
            cv2.circle(image_display, (int(x), int(y)), 5, (0, 255, 0), -1)
            label = f"{label_prefix}{len(points)}" if label_prefix else f"{len(points)}"
            cv2.putText(
                image_display,
                label,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow(window_name, image_display)

    cv2.setMouseCallback(window_name, _callback)


def wait_for_enter_or_esc() -> str:
    """Block until Enter or ESC is pressed; return 'enter' or 'esc'."""
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            return "enter"
        if key == 27:  # ESC
            return "esc"


# ----------------------- Main annotation flow -----------------------


def annotate_images(image_folder: str, output_csv: str) -> None:
    """Iterate over images in folder, collect path (optional) + trees, save
    CSV."""
    image_list = sorted(f for f in os.listdir(image_folder) if f.lower().endswith(".jpg"))

    window_name = "SGIL Annotation"
    init_window(window_name, x=250, y=250)

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_filename", "num_trees", "tree_points", "path_points"])

        for image_name in image_list:
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[warn] Could not load image: {image_name}")
                continue

            print(f"\nImage: {image_name}")

            # -------- Stage 1: Optional path selection (4 points: BL, TL, BR, TR) --------
            path_points: list[Point] = []
            path_display = image.copy()
            cv2.imshow(window_name, path_display)
            cv2.waitKey(1)

            print("Path selection:")
            print("  Click 4 corners in order: BL, TL, BR, TR.")
            print("  Press Enter to finish (press Enter immediately if no path is visible).")
            set_click_callback(window_name, path_display, path_points, label_prefix="P")

            action = wait_for_enter_or_esc()
            if action == "esc":
                # Skip image entirely
                print("  Skipping image.")
                cv2.setMouseCallback(window_name, lambda *args: None)
                continue

            # Freeze callbacks before switching stages
            cv2.setMouseCallback(window_name, lambda *args: None)

            # Convert path data to CSV-friendly format
            if len(path_points) == 4:
                path_points_csv: str | list[tuple[int, int]] = [
                    (int(p.x), int(p.y)) for p in path_points
                ]
            else:
                path_points_csv = "NO PATH"

            # -------- Stage 2: Tree selection (any number) --------
            tree_points: list[Point] = []
            trees_display = image.copy()
            cv3 = window_name  # alias to emphasize reuse
            cv2.imshow(cv3, trees_display)
            cv2.waitKey(1)

            print("Tree selection:")
            print("  Left-click each tree to add a point.")
            print("  Press Enter when done, or ESC to skip trees for this image.")
            set_click_callback(cv3, trees_display, tree_points, label_prefix="T")

            action = wait_for_enter_or_esc()
            if action == "esc":
                tree_points = []

            # Freeze callback for safety
            cv2.setMouseCallback(window_name, lambda *args: None)

            # Save row
            num_trees = len(tree_points)
            tree_points_csv = [(int(p.x), int(p.y)) for p in tree_points]
            writer.writerow([image_name, num_trees, tree_points_csv, path_points_csv])

            print(
                f"  Saved {num_trees} tree(s). Path: {path_points_csv if isinstance(path_points_csv, str) else '4 points'}"
            )

    print(f"\nAll done! Results saved to {output_csv}")


# ----------------------- Entrypoint -----------------------

if __name__ == "__main__":
    adjusted_image_folder_path = "../" + IMAGE_FOLDER_PATH
    if not os.path.exists(adjusted_image_folder_path):
        print(f"Image folder not found: {adjusted_image_folder_path}")
        sys.exit(1)

    annotate_images(adjusted_image_folder_path, OUTPUT_CSV)
