"""Download a range of images from the remote machine to *your local computer*.

This repo lives on a machine you're SSH'd into. To copy files *to your computer*,
run this script on your local machine (the one with the destination folder).

It uses `scp` under the hood, so you need OpenSSH client tools installed and
SSH access configured.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REMOTE = "jgelia@wolfwagen-ub01.csc.ncsu.edu"

# Absolute path on the remote machine to the images directory.
REMOTE_IMAGES_DIR = "/mnt/sdb/DowntownRaleigh/images"

# Where to put the downloaded images on your *local* machine.
# If relative, it's relative to where you run this script.
LOCAL_DEST_DIR = "./downloaded_images"

# Inclusive index range to download.
START_INDEX = 0
END_INDEX = 100

# Download every Nth image. Set to 10 to download every 10th image.
STEP = 10

# Your password constant. NOTE: Requires 'sshpass' to be installed on your machine.
SSH_PASSWORD = "your_actual_password_here"

# Image naming scheme.
FILENAME_TEMPLATE = "image_{index:08d}.png"

# Optional SSH settings.
SSH_PORT: int | None = None
SSH_IDENTITY_FILE: str | None = None  # e.g. "~/.ssh/id_ed25519"

# Optional SSH control socket to reuse a single authenticated connection.
# Example Windows path: r"C:\Users\you\.ssh\scp_ctrl"
SSH_CONTROL_PATH: str | None = None
SSH_CONTROL_PERSIST: str | None = None  # e.g. "5m"

# If True, prints the scp commands without running them.
DRY_RUN = False


def _expand_user_path(path_str: str | None) -> str | None:
    if not path_str:
       return None
    return str(Path(path_str).expanduser())


def _scp_base_args() -> list[str]:
    args = ["scp", "-p"]

    if SSH_PORT is not None:
       args.extend(["-P", str(SSH_PORT)])

    identity_file = _expand_user_path(SSH_IDENTITY_FILE)
    if identity_file:
       args.extend(["-i", identity_file])

    control_path = _expand_user_path(SSH_CONTROL_PATH)
    if control_path:
       args.extend(["-o", "ControlMaster=auto", "-o", f"ControlPath={control_path}"])
       if SSH_CONTROL_PERSIST:
          args.extend(["-o", f"ControlPersist={SSH_CONTROL_PERSIST}"])

    return args


def _remote_image_path(index: int) -> str:
    filename = FILENAME_TEMPLATE.format(index=index)
    return f"{REMOTE}:{REMOTE_IMAGES_DIR.rstrip('/')}/{filename}"


def _local_image_path(dest_dir: Path, index: int) -> Path:
    filename = FILENAME_TEMPLATE.format(index=index)
    return dest_dir / filename


def main() -> int:
    if START_INDEX < 0:
       print("START_INDEX must be >= 0", file=sys.stderr)
       return 2
    if END_INDEX < START_INDEX:
       print("END_INDEX must be >= START_INDEX", file=sys.stderr)
       return 2
    if STEP <= 0:
       print("STEP must be >= 1", file=sys.stderr)
       return 2

    if REMOTE == "you@your-remote-host":
       print(
          "Edit REMOTE at the top of download_images.py (e.g. 'you@host' or SSH alias).",
          file=sys.stderr,
       )
       return 2

    dest_dir = Path(LOCAL_DEST_DIR).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    scp_base = _scp_base_args()
    failures: list[int] = []

    indices = list(range(START_INDEX, END_INDEX + 1, STEP))
    total = len(indices)
    print(f"Downloading {total} images to {dest_dir} ...")

    for i, index in enumerate(indices, start=1):
       remote_path = _remote_image_path(index)
       local_path = _local_image_path(dest_dir, index)

       cmd = scp_base + [remote_path, str(local_path)]

       # Prepend sshpass if a password is provided
       if SSH_PASSWORD:
           cmd = ["sshpass", "-p", SSH_PASSWORD] + cmd

       if DRY_RUN:
          print(" ".join(cmd))
          continue

       proc = subprocess.run(cmd, check=False)
       if proc.returncode != 0:
          failures.append(index)
          print(f"[{i}/{total}] FAILED: {remote_path}", file=sys.stderr)
       else:
          print(f"[{i}/{total}] OK: {local_path.name}")

    if failures:
       joined = ", ".join(map(str, failures[:25]))
       more = "" if len(failures) <= 25 else f" (+{len(failures) - 25} more)"
       print(f"Done with failures at indices: {joined}{more}", file=sys.stderr)
       return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())