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

from PIL import Image

REMOTE = "jgelia@wolfwagen-ub01.csc.ncsu.edu"

# Absolute path on the remote machine to the images directory.
REMOTE_IMAGES_DIR = "/mnt/sdb/DowntownRaleigh/images"

# Where to put the downloaded images on your *local* machine.
# If relative, it's relative to where you run this script.
LOCAL_DEST_DIR = "../dataset/raleigh_images"


# ==========================================
# --- DOWNLOAD SELECTION MODE ---
# ==========================================

# Set to True to download exactly the indices in SPECIFIC_INDICES.
# Set to False to use START_INDEX, END_INDEX, and STEP.
USE_SPECIFIC_INDICES = True

# Explicit list of image indices to download.
# (Used only if USE_SPECIFIC_INDICES is True)
SPECIFIC_INDICES: list[int] = [140, 152, 167, 182, 198]

# Inclusive index range to download.
# (Used only if USE_SPECIFIC_INDICES is False)
START_INDEX = 140
END_INDEX = 240

# Download every Nth image (10 = every 10th image).
# (Used only if USE_SPECIFIC_INDICES is False)
STEP = 10


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
    # Validate based on the selected mode
    if USE_SPECIFIC_INDICES:
        if not SPECIFIC_INDICES:
            print("SPECIFIC_INDICES cannot be empty when USE_SPECIFIC_INDICES is True.", file=sys.stderr)
            return 2
    else:
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

    # Generate the target indices based on mode
    if USE_SPECIFIC_INDICES:
        indices = SPECIFIC_INDICES
    else:
        indices = list(range(START_INDEX, END_INDEX + 1, STEP))

    total = len(indices)
    print(f"Downloading {total} images to {dest_dir} ...")

    for i, index in enumerate(indices, start=1):
       remote_path = _remote_image_path(index)
       local_path = _local_image_path(dest_dir, index)

       cmd = scp_base + [remote_path, str(local_path)]

       if DRY_RUN:
          print(" ".join(cmd))
          continue

       proc = subprocess.run(cmd, check=False)
       if proc.returncode != 0:
          failures.append(index)
          print(f"[{i}/{total}] FAILED: {remote_path}", file=sys.stderr)
       else:
          print(f"[{i}/{total}] OK: {local_path.name}")

          # --- NEW CROP LOGIC ---
          try:
              with Image.open(local_path) as img:
                  width, height = img.size
                  # The crop box is defined as a tuple: (left, top, right, bottom)
                  # So we go from X=0 to X=width//2 (the middle)
                  left_half = img.crop((0, 0, width // 2, height))
                  left_half.save(local_path)
              print(f"[{i}/{total}] CROPPED to left half: {local_path.name}")
          except Exception as e:
              print(f"[{i}/{total}] CROP FAILED for {local_path.name}: {e}", file=sys.stderr)

    if failures:
       joined = ", ".join(map(str, failures[:25]))
       more = "" if len(failures) <= 25 else f" (+{len(failures) - 25} more)"
       print(f"Done with failures at indices: {joined}{more}", file=sys.stderr)
       return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
