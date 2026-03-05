# Project SGIL

The goal of this project is to use CV paired with satellite imagery to improve localization of a cheap GPS.

## Setup

This project is for Linux systems. If on Windows, please use WSL. Also, depending on the images you're using, you will need to download them separately.

### Setup Environment

We use uv for our project. You can install it [here](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Setup venv with uv
uv sync --all-extras
```

### Lint and format the code

```bash
uv run ruff format .
uv run ruff check . --fix
```

### Run the code

```bash
# While in the root of the project, run:
uv run python -m project_sgil.manual_sgil
```

## Manual tree labeling

Use the manual labeling tool to click tree centers on the Raleigh satellite image. Clicks are saved to a CSV next to the image and existing points are reloaded when you reopen the tool.

The CSV format is two columns: `lat,lon` with no header.

```bash
# Run the labeling tool
uv run python scripts\manual_sat_tree_detection.py
```

### Convert old pixel CSV to lat/lon

If you have an older pixel-based CSV, run:

```bash
uv run python scripts\convert_tree_csv_to_latlon.py
```

```
uv run python -m project_sgil.automated_sgil | Out-File -FilePath "dataset\tables\automated_sgil_table.txt" -Encoding utf8
uv run python -m project_sgil.graphics.sgil_visualizer "dataset\tables\automated_sgil_table.txt" --out "plot.png"
```

This overwrites `RaleighSatellite_manual_trees.csv` with lat/lon values and writes a backup at `RaleighSatellite_manual_trees.csv.bak`.
