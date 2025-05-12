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

### Install pre-commit

```bash
pre-commit install
```

## Usage

We use pre‑commit to automatically run Ruff and Docformatter on every commit.

### Lint and format the code

```bash
pre-commit run --all-files
```