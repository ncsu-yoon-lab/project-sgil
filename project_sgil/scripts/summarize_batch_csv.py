"""Summarize errors for a batch_results-style CSV.

For each estimate set (columns like "<tag>__x" and "<tag>__y"), computes the
Euclidean error to RTK (rtk_x, rtk_y) and prints:
  - mean
  - median
  - std deviation

It prints these stats for:
  1) all rows
  2) rows excluding the no-tree zone (image indices 303..590 inclusive)

Usage:
    uv run ./scripts/summarize_batch_csv.py ../batch_results.csv

Notes:
  - A row is counted for a given estimate set only if rtk_x/rtk_y and that set's
    x/y are all finite.
  - The no-tree zone filter is based on the numeric index embedded in the
    "image" column (e.g. images/image_00000303.png).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


NO_TREE_START = 303
NO_TREE_END = 590


@dataclass(frozen=True)
class Stats:
    n: int
    mean: float
    median: float
    std: float


def _extract_image_index(image_name: str) -> int | None:
    """Extract numeric index from an image filename."""
    if not isinstance(image_name, str):
        return None
    m = re.search(r"(\d+)", image_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _no_tree_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows inside the no-tree zone."""
    if "image" not in df.columns:
        return pd.Series(False, index=df.index)

    idxs = df["image"].apply(_extract_image_index)
    return idxs.notna() & (idxs >= NO_TREE_START) & (idxs <= NO_TREE_END)


def _find_estimate_sets(df: pd.DataFrame) -> list[str]:
    """Return list of estimate set tags based on columns '<tag>__x' and '<tag>__y'."""
    x_cols = [c for c in df.columns if c.endswith("__x")]
    tags: list[str] = []
    for xc in x_cols:
        tag = xc[:-3]
        yc = f"{tag}__y"
        if yc in df.columns:
            tags.append(tag)
    return sorted(tags)


def _compute_stats(errors: np.ndarray) -> Stats | None:
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return None
    return Stats(
        n=int(errors.size),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors, ddof=0)),
    )


def _errors_for_tag(df: pd.DataFrame, tag: str) -> np.ndarray:
    """Compute per-row euclidean error for a given tag against RTK."""
    x = df[f"{tag}__x"].to_numpy(dtype=float, copy=False)
    y = df[f"{tag}__y"].to_numpy(dtype=float, copy=False)
    rx = df["rtk_x"].to_numpy(dtype=float, copy=False)
    ry = df["rtk_y"].to_numpy(dtype=float, copy=False)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(rx) & np.isfinite(ry)
    if not np.any(ok):
        return np.array([], dtype=float)

    dx = x[ok] - rx[ok]
    dy = y[ok] - ry[ok]
    return np.hypot(dx, dy)


def _errors_for_xy(df: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    """Compute per-row euclidean error for a given (x_col, y_col) against RTK."""
    if x_col not in df.columns or y_col not in df.columns:
        return np.array([], dtype=float)

    x = df[x_col].to_numpy(dtype=float, copy=False)
    y = df[y_col].to_numpy(dtype=float, copy=False)
    rx = df["rtk_x"].to_numpy(dtype=float, copy=False)
    ry = df["rtk_y"].to_numpy(dtype=float, copy=False)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(rx) & np.isfinite(ry)
    if not np.any(ok):
        return np.array([], dtype=float)

    dx = x[ok] - rx[ok]
    dy = y[ok] - ry[ok]
    return np.hypot(dx, dy)


def _print_table(header: str, rows: list[tuple[str, Stats | None]]) -> None:
    print("\n" + header)
    print("=" * len(header))
    print(f"{'tag':70s} | {'n':>6s} | {'mean':>10s} | {'median':>10s} | {'std':>10s}")
    print("-" * 70 + "-+-" + "-" * 6 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10)

    for tag, st in rows:
        if st is None:
            print(f"{tag:70s} | {'0':>6s} | {'nan':>10s} | {'nan':>10s} | {'nan':>10s}")
        else:
            print(
                f"{tag:70s} | {st.n:6d} | {st.mean:10.3f} | {st.median:10.3f} | {st.std:10.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize batch_results CSV")
    ap.add_argument("csv", help="Path to batch_results-style CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"rtk_x", "rtk_y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    zone_mask = _no_tree_mask(df)

    # ---- GPS summary ----
    gps_rows_all: list[tuple[str, Stats | None]] = [
        ("gps", _compute_stats(_errors_for_xy(df, "gps_x", "gps_y")))
    ]
    df_non_zone = df.loc[~zone_mask]
    gps_rows_nz: list[tuple[str, Stats | None]] = [
        ("gps", _compute_stats(_errors_for_xy(df_non_zone, "gps_x", "gps_y")))
    ]

    _print_table("GPS error (vs RTK) — all rows", gps_rows_all)
    _print_table(
        f"GPS error (vs RTK) — excluding no-tree zone ({NO_TREE_START}..{NO_TREE_END})",
        gps_rows_nz,
    )

    # ---- Estimate-set summaries ----
    tags = _find_estimate_sets(df)
    if not tags:
        raise SystemExit(
            "No estimate sets found (expected columns like '<tag>__x' and "
            "'<tag>__y')."
        )

    all_rows: list[tuple[str, Stats | None]] = []
    non_zone_rows: list[tuple[str, Stats | None]] = []

    for tag in tags:
        errs_all = _errors_for_tag(df, tag)
        all_rows.append((tag, _compute_stats(errs_all)))

        errs_nz = _errors_for_tag(df_non_zone, tag)
        non_zone_rows.append((tag, _compute_stats(errs_nz)))

    _print_table("All rows", all_rows)
    _print_table(
        f"Rows excluding no-tree zone ({NO_TREE_START}..{NO_TREE_END} inclusive)",
        non_zone_rows,
    )


if __name__ == "__main__":
    main()
