import math
import os

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import *

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting


class GPSRTKVisualizer:
    def __init__(self, data_path, save_path=None) -> None:
        """
        Initialize the visualizer with path to the data file.

        Args:
            data_path (str): Path to the CSV file containing GPS and RTK data
            save_path (str, optional): Path to save the visualization images
        """
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.current_index = 0
        self.fig = None
        self.ax = None
        self.rtk_scatter = None
        self.gps_scatter = None
        self.current_rtk_point = None
        self.current_gps_point = None
        self.rtk_line = None
        self.gps_line = None
        self.fig_initialized = False

    def load_data(self) -> None:
        """Load the CSV data and prepare it for visualization."""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)

        # Filter out rows with NaN values in lat/lon columns
        self.data = self.data.dropna(subset=["rtk_lat", "rtk_lon", "gps_lat", "gps_lon"])

        # Reset index after dropping rows
        self.data = self.data.reset_index(drop=True)

        print(f"Loaded {len(self.data)} valid data points")

        # Sort by timestamp if it exists
        if "timestamp" in self.data.columns:
            self.data = self.data.sort_values("timestamp").reset_index(drop=True)
            print("Data sorted by timestamp")

    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on the earth
        (specified in decimal degrees).
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r

    def initialize_plot(self) -> None:
        """Set up the initial plot."""
        if self.fig_initialized:
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # Plot all RTK points
        rtk_lats = self.data["rtk_lat"].values
        rtk_lons = self.data["rtk_lon"].values
        self.rtk_scatter = self.ax.scatter(
            rtk_lons, rtk_lats, color="red", alpha=0.5, label="RTK Path"
        )

        # Plot all GPS points
        gps_lats = self.data["gps_lat"].values
        gps_lons = self.data["gps_lon"].values
        self.gps_scatter = self.ax.scatter(
            gps_lons, gps_lats, color="blue", alpha=0.5, label="GPS Path"
        )

        # Initialize lines for the paths
        (self.rtk_line,) = self.ax.plot([], [], "r-", linewidth=1)
        (self.gps_line,) = self.ax.plot([], [], "b-", linewidth=1)

        # Initialize current points
        self.current_rtk_point = self.ax.scatter(
            [], [], color="darkred", s=100, label="Current RTK"
        )
        self.current_gps_point = self.ax.scatter(
            [], [], color="darkblue", s=100, label="Current GPS"
        )

        # Set axis labels and title
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("GPS vs RTK Position Comparison")

        # Add legend
        self.ax.legend()

        # Adjust plot to fit data
        self.ax.grid(True)

        # Set equal aspect to prevent distortion
        self.ax.set_aspect("equal", adjustable="box")

        self.fig_initialized = True

        # Setup key press event
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Show the initial plot
        plt.tight_layout()
        plt.show(block=False)

    def update_plot(self) -> bool:
        """Update the plot with the current data point."""
        if self.current_index >= len(self.data):
            print("Reached the end of the data")
            return False

        # Get current data point
        current_data = self.data.iloc[self.current_index]

        # Update RTK point
        rtk_lat = current_data["rtk_lat"]
        rtk_lon = current_data["rtk_lon"]
        self.current_rtk_point.set_offsets(np.array([[rtk_lon, rtk_lat]]))

        # Update GPS point
        gps_lat = current_data["gps_lat"]
        gps_lon = current_data["gps_lon"]
        self.current_gps_point.set_offsets(np.array([[gps_lon, gps_lat]]))

        # Update lines (paths)
        rtk_lons = self.data["rtk_lon"].iloc[: self.current_index + 1].values
        rtk_lats = self.data["rtk_lat"].iloc[: self.current_index + 1].values
        self.rtk_line.set_data(rtk_lons, rtk_lats)

        gps_lons = self.data["gps_lon"].iloc[: self.current_index + 1].values
        gps_lats = self.data["gps_lat"].iloc[: self.current_index + 1].values
        self.gps_line.set_data(gps_lons, gps_lats)

        # Calculate error between RTK and GPS
        error_m = self.calculate_haversine_distance(rtk_lat, rtk_lon, gps_lat, gps_lon)

        # Update title with error information
        timestamp_info = (
            f"Time: {current_data['timestamp']:.2f}" if "timestamp" in current_data else ""
        )
        self.ax.set_title(
            f"GPS vs RTK Position (Point {self.current_index + 1}/{len(self.data)}) - Error: {error_m:.2f}m {timestamp_info}"
        )

        # Update the canvas
        self.fig.canvas.draw_idle()

        # Save the figure if save_path is specified
        if self.save_path:
            save_file = os.path.join(self.save_path, f"frame_{self.current_index:04d}.png")
            self.fig.savefig(save_file)
            print(f"Saved frame to {save_file}")

        return True

    def on_key_press(self, event) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.next_point()
        elif event.key == "backspace":
            self.previous_point()
        elif event.key == "escape":
            plt.close(self.fig)

    def next_point(self) -> None:
        """Advance to the next data point."""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.update_plot()
        else:
            print("Reached the end of the data")

    def previous_point(self) -> None:
        """Go back to the previous data point."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()
        else:
            print("Already at the beginning of the data")

    def create_interactive_map(self, output_file="gps_rtk_map.html") -> None:
        """Create an interactive folium map with the data."""
        # Calculate the center of the data
        center_lat = self.data["rtk_lat"].mean()
        center_lon = self.data["rtk_lon"].mean()

        # Create a map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

        # Add RTK path
        rtk_coords = [
            (lat, lon) for lat, lon in zip(self.data["rtk_lat"], self.data["rtk_lon"], strict=False)
        ]
        folium.PolyLine(rtk_coords, color="red", weight=3, opacity=0.7, tooltip="RTK Path").add_to(
            m
        )

        # Add GPS path
        gps_coords = [
            (lat, lon) for lat, lon in zip(self.data["gps_lat"], self.data["gps_lon"], strict=False)
        ]
        folium.PolyLine(gps_coords, color="blue", weight=3, opacity=0.7, tooltip="GPS Path").add_to(
            m
        )

        # Add markers for each point with popups showing the error
        for i, row in self.data.iterrows():
            # Calculate error
            error_m = self.calculate_haversine_distance(
                row["rtk_lat"], row["rtk_lon"], row["gps_lat"], row["gps_lon"]
            )

            # Add RTK marker
            rtk_popup_text = (
                f"RTK Point {i + 1}<br>Lat: {row['rtk_lat']:.8f}<br>Lon: {row['rtk_lon']:.8f}"
            )
            if "timestamp" in row:
                rtk_popup_text += f"<br>Time: {row['timestamp']:.2f}"
            folium.CircleMarker(
                location=[row["rtk_lat"], row["rtk_lon"]],
                radius=3,
                color="darkred",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
                popup=rtk_popup_text,
            ).add_to(m)

            # Add GPS marker
            gps_popup_text = f"GPS Point {i + 1}<br>Lat: {row['gps_lat']:.8f}<br>Lon: {row['gps_lon']:.8f}<br>Error from RTK: {error_m:.2f}m"
            if "timestamp" in row:
                gps_popup_text += f"<br>Time: {row['timestamp']:.2f}"
            folium.CircleMarker(
                location=[row["gps_lat"], row["gps_lon"]],
                radius=3,
                color="darkblue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.7,
                popup=gps_popup_text,
            ).add_to(m)

            # Draw a line between corresponding RTK and GPS points
            folium.PolyLine(
                [[row["rtk_lat"], row["rtk_lon"]], [row["gps_lat"], row["gps_lon"]]],
                color="green",
                weight=1,
                opacity=0.5,
                dash_array="5, 5",
                tooltip=f"Error: {error_m:.2f}m",
            ).add_to(m)

        # Save the map
        m.save(output_file)
        print(f"Interactive map saved to {output_file}")

    def run(self) -> None:
        """Run the visualization."""
        # Load the data
        self.load_data()

        if len(self.data) == 0:
            print("No valid data points found")
            return

        # Initialize the plot
        self.initialize_plot()

        # Update with the first point
        self.update_plot()

        # Keep the plot open
        plt.show()

        print(
            "Visualization complete. Press Enter to advance, Backspace to go back, or Escape to close."
        )


if __name__ == "__main__":
    # Create the visualizer with the path to your data
    visualizer = GPSRTKVisualizer(DATA_LOGGER_PATH)  # Replace with your CSV file path

    # Run the visualization
    visualizer.run()

    # Optionally create an interactive HTML map
    visualizer.create_interactive_map()
