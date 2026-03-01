from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


POSE_RE = re.compile(
    r"\(\s*x\s*=\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"y\s*=\s*(?P<y>[-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"yaw\s*=\s*(?P<yaw>[-+]?\d+(?:\.\d+)?)\s*°\s*\)"
)


@dataclass
class Row:
    frame: str
    matched: Optional[bool]
    sgil_err_m: Optional[float]
    gps_err_m: Optional[float]
    poses: Dict[str, Optional[Tuple[float, float, float]]]  # name -> (x, y, yaw_deg)


def parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_bool(s: str) -> Optional[bool]:
    s = (s or "").strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def parse_pose(cell: str) -> Optional[Tuple[float, float, float]]:
    cell = (cell or "").strip()
    if not cell:
        return None
    m = POSE_RE.search(cell)
    if not m:
        return None
    return (float(m.group("x")), float(m.group("y")), float(m.group("yaw")))


def parse_table_txt(path: Path) -> List[Row]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    data_lines: List[str] = []
    for ln in lines:
        if "|" not in ln:
            continue
        if set(ln.strip()) <= set("-+"):
            continue
        if "frame" in ln and "sgil" in ln and "gps" in ln and "matched" in ln:
            continue
        if not ln.strip():
            continue
        data_lines.append(ln)

    rows: List[Row] = []
    for ln in data_lines:
        parts = [p.strip() for p in ln.split("|")]

        # frame | matched | sgil_err_m | gps_err_m | gps_pose | rtk_pose | current_pose | estimated_pose
        if len(parts) < 8:
            continue

        frame = parts[0]
        matched = parse_bool(parts[1])
        sgil_err_m = parse_float(parts[2])
        gps_err_m = parse_float(parts[3])

        poses = {
            "gps_pose": parse_pose(parts[4]),
            "rtk_pose": parse_pose(parts[5]),
            "current_pose": parse_pose(parts[6]),
            "estimated_pose": parse_pose(parts[7]),
        }

        rows.append(Row(frame=frame, matched=matched, sgil_err_m=sgil_err_m, gps_err_m=gps_err_m, poses=poses))

    if not rows:
        raise ValueError("No data rows parsed. Check that the file is pipe-delimited like the example.")
    return rows


def plot_rows(
    rows: List[Row],
    out_path: Path,
    title: Optional[str],
    draw_headings: bool,
    include_error_plot: bool,
    heading_stride: int,
    heading_len_m: float,
) -> None:
    names = ["gps_pose", "rtk_pose", "current_pose", "estimated_pose"]

    xy: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for n in names:
        xs: List[float] = []
        ys: List[float] = []
        yaws: List[float] = []
        for r in rows:
            p = r.poses.get(n)
            if p is None:
                xs.append(float("nan"))
                ys.append(float("nan"))
                yaws.append(float("nan"))
            else:
                xs.append(p[0]); ys.append(p[1]); yaws.append(p[2])
        xy[n] = (xs, ys, yaws)

    idx = list(range(len(rows)))
    sgil_err = [r.sgil_err_m if r.sgil_err_m is not None else float("nan") for r in rows]
    gps_err = [r.gps_err_m if r.gps_err_m is not None else float("nan") for r in rows]

    if include_error_plot:
        fig = plt.figure(figsize=(12, 7), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_err = fig.add_subplot(gs[1, 0])
    else:
        fig, ax_xy = plt.subplots(figsize=(12, 7), dpi=150)
        ax_err = None

    # XY trajectories
    for n in names:
        xs, ys, _ = xy[n]
        ax_xy.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=n)

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")
    ax_xy.set_title(title if title else f"Trajectories from {out_path.name}")

    # Heading arrows (optional)
    if draw_headings:
        import math
        for n in names:
            xs, ys, yaws = xy[n]
            for i in range(0, len(rows), max(1, heading_stride)):
                x, y, yaw = xs[i], ys[i], yaws[i]
                if not (x == x and y == y and yaw == yaw):  # NaN check
                    continue
                theta = math.radians(yaw)
                dx = heading_len_m * math.cos(theta)
                dy = heading_len_m * math.sin(theta)
                ax_xy.arrow(
                    x, y, dx, dy,
                    length_includes_head=True,
                    head_width=heading_len_m * 0.20,
                    head_length=heading_len_m * 0.25,
                    alpha=0.35,
                    linewidth=0.8,
                )

    # Error plot + stars for every matched=False point (on sgil_err series)
    if include_error_plot and ax_err is not None:
        ax_err.plot(idx, sgil_err, marker="o", linewidth=1.2, markersize=3, label="sgil_err_m")
        ax_err.plot(idx, gps_err, marker="o", linewidth=1.2, markersize=3, label="gps_err_m")

        star_x: List[int] = []
        star_y: List[float] = []
        for i, r in enumerate(rows):
            if r.matched is False and r.sgil_err_m is not None:
                star_x.append(i)
                star_y.append(float(r.sgil_err_m))

        if star_x:
            ax_err.scatter(
                star_x,
                star_y,
                marker="*",
                s=140,
                linewidths=0.8,
                label="matched=False (sgil_err)",
            )

        ax_err.set_xlabel("Row index")
        ax_err.set_ylabel("Error (m)")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", type=Path, help="Path to the pipe-delimited .txt table")
    ap.add_argument("--out", type=Path, default=Path("plot.png"), help="Output PNG path")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title")
    ap.add_argument("--no_headings", action="store_true", help="Disable heading arrows")
    ap.add_argument("--no_errors", action="store_true", help="Disable sgil/gps error subplot")
    ap.add_argument("--heading_stride", type=int, default=3, help="Draw a heading arrow every N rows")
    ap.add_argument("--heading_len_m", type=float, default=0.8, help="Heading arrow length in meters")
    args = ap.parse_args()

    rows = parse_table_txt(args.input_txt)
    plot_rows(
        rows=rows,
        out_path=args.out,
        title=args.title,
        draw_headings=not args.no_headings,
        include_error_plot=not args.no_errors,
        heading_stride=max(1, args.heading_stride),
        heading_len_m=max(0.01, args.heading_len_m),
    )
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()