# Project SGIL

The goal of this project is to use CV paired with satellite imagery to improve localization of a cheap GPS.

## Setup

This project is for Linux systems. If on Windows, please use WSL.

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
uv run 