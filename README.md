# Project SGIL

The goal of this project is to use CV paired with satellite imagery to improve localization of a cheap GPS.

## Setup

This project is for Linux systems. If on Windows, please use WSL.

### Install system prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

## Usage

We use Ruff as our linter and formatter to keep the codebase clean and consistent.

### Lint and format the code

```bash
ruff check . --fix
ruff format .
```