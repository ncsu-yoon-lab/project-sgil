import csv
import os
import sys
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# ============================================================
# Manual tree center labeling on the satellite image
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from project_sgil.constants import IMAGE_BOTTOM_RIGHT, IMAGE_TOP_LEFT

# ============================================================
# Manual tree center labeling on the satellite image
# ============================================================

IMAGE_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../..", "dataset", "satellite_trees", "RaleighSatellite.png")
)
CSV_PATH = os.path.join(os.path.dirname(IMAGE_PATH), "RaleighSatellite_manual_trees.csv")

POINT_COLOR = "lime"
POINT_SIZE = 20
ZOOM_BASE_SCALE = 1.2
REMOVE_RADIUS_PX = 12
SAVE_EVERY_N = 10


def _load_points(csv_path: str) -> List[Tuple[float, float]]:
    if not os.path.exists(csv_path):
        return []

    points: List[Tuple[float, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                lat = float(row[0])
                lon = float(row[1])
            except ValueError:
                continue
            points.append((lat, lon))

    return points


def _save_points(csv_path: str, points: List[Tuple[float, float]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(points)


class TreeClicker:
    def __init__(self, img_path: str, csv_path: str) -> None:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        self.img_path = img_path
        self.csv_path = csv_path

        self.img = mpimg.imread(img_path)
        self.image_height = int(self.img.shape[0])
        self.image_width = int(self.img.shape[1])

        self.lat_top = float(IMAGE_TOP_LEFT.x)
        self.lon_left = float(IMAGE_TOP_LEFT.y)
        self.lat_bottom = float(IMAGE_BOTTOM_RIGHT.x)
        self.lon_right = float(IMAGE_BOTTOM_RIGHT.y)

        self.points = _load_points(csv_path)
        self._dirty = False
        self._ops_since_save = 0

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, interpolation="nearest")
        self._set_title()

        self.scatter = self.ax.scatter(
            [p[0] for p in self._points_pixel()],
            [p[1] for p in self._points_pixel()],
            s=POINT_SIZE,
            c=POINT_COLOR,
            marker="o",
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _set_title(self, *, lat: float | None = None, lon: float | None = None) -> None:
        if lat is None or lon is None:
            title = (
                "Click tree centers (left-click). Right-click to remove. Scroll to zoom."
            )
        else:
            title = (
                "Click tree centers (left-click). Right-click to remove. Scroll to zoom. "
                f"Last: {lat:.6f}, {lon:.6f}"
            )
        self.ax.set_title(title)

    def _pixel_to_latlon(self, x_px: float, y_px: float) -> Tuple[float, float]:
        if self.image_width <= 1 or self.image_height <= 1:
            return self.lat_top, self.lon_left
        lon = self.lon_left + (x_px / (self.image_width - 1)) * (self.lon_right - self.lon_left)
        lat = self.lat_top + (y_px / (self.image_height - 1)) * (self.lat_bottom - self.lat_top)
        return lat, lon

    def _latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int] | None:
        if self.image_width <= 1 or self.image_height <= 1:
            return None
        if self.lon_right == self.lon_left or self.lat_bottom == self.lat_top:
            return None
        x = (lon - self.lon_left) / (self.lon_right - self.lon_left) * (self.image_width - 1)
        y = (lat - self.lat_top) / (self.lat_bottom - self.lat_top) * (self.image_height - 1)
        x_int = int(round(x))
        y_int = int(round(y))
        if not (0 <= x_int < self.image_width and 0 <= y_int < self.image_height):
            return None
        return x_int, y_int

    def _points_pixel(self) -> List[Tuple[int, int]]:
        pixels: List[Tuple[int, int]] = []
        for lat, lon in self.points:
            pixel = self._latlon_to_pixel(lat, lon)
            if pixel is not None:
                pixels.append(pixel)
        return pixels

    def _save_if_needed(self, *, force: bool = False) -> None:
        if force or (self._dirty and self._ops_since_save >= SAVE_EVERY_N):
            _save_points(self.csv_path, self.points)
            self._dirty = False
            self._ops_since_save = 0

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                return
            lat, lon = self._pixel_to_latlon(x, y)
            self.points.append((lat, lon))
            self._dirty = True
            self._ops_since_save += 1
            self._set_title(lat=lat, lon=lon)
            pixels = self._points_pixel()
            self.scatter.set_offsets(pixels)
            self._save_if_needed()
            self.fig.canvas.draw_idle()
            return

        if event.button == 3:
            if self._remove_nearest_point(event.xdata, event.ydata):
                self._dirty = True
                self._ops_since_save += 1
                pixels = self._points_pixel()
                self.scatter.set_offsets(pixels)
                self._save_if_needed()
                self.fig.canvas.draw_idle()

    def _remove_nearest_point(self, x: float, y: float) -> bool:
        if not self.points:
            return False

        pixels = self._points_pixel()
        if not pixels:
            return False

        best_idx = -1
        best_dist2 = float("inf")
        for idx, (px, py) in enumerate(pixels):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = idx

        if best_idx >= 0 and best_dist2 <= (REMOVE_RADIUS_PX * REMOVE_RADIUS_PX):
            self.points.pop(best_idx)
            return True

        return False

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if event.button == "up":
            scale_factor = 1 / ZOOM_BASE_SCALE
        elif event.button == "down":
            scale_factor = ZOOM_BASE_SCALE
        else:
            scale_factor = 1.0

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.fig.canvas.draw_idle()

    def _on_close(self, event) -> None:
        self._save_if_needed(force=True)

    def run(self) -> None:
        plt.show()


def main() -> None:
    clicker = TreeClicker(IMAGE_PATH, CSV_PATH)
    clicker.run()


if __name__ == "__main__":
    main()

