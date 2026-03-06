"""Grid search for Automated SGIL configuration.

Runs AutomatedSGIL repeatedly while varying a small set of constants and
reports the configuration with the lowest mean SGIL error.

This script modifies `project_sgil.constants` *in-memory only* (no file edits)
per run. It should be run from the repo root so `project_sgil` is importable.

Search space (inclusive):
- HEADING_ERROR_DEG: 2..5 step 1
- HEADING_SWEEP_RANGE_DEG: 0.5..2.5 step 0.5
- MAX_ESTIMATE_DIST_FROM_SNAPPED_POSITION_SWEEP: 3..5 step 1
- AOI_RADIUS_M: 65..80 step 5

Outputs:
- prints per-combo metrics
- writes `grid_search_automated_sgil_configs_results.csv` in the repo root

Note: AutomatedSGIL/TreeMatcher import constants as module-level values, so we
reload those modules after patching constants each iteration.
"""

from __future__ import annotations

import csv
import importlib
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass

# Ensure repo root is on sys.path so `import project_sgil` works when running
# this file directly from the `scripts/` directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass(frozen=True)
class Combo:
    heading_error_deg: int
    heading_sweep_range_deg: float
    max_estimate_dist_from_snapped_position_sweep: int
    aoi_radius_m: int


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range with stable rounding."""
    vals: list[float] = []
    x = start
    # Use integer steps to avoid drift
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        vals.append(round(start + i * step, 10))
    return vals


def _mean_finite(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _evaluate_combo(combo: Combo) -> dict[str, float | int | str | None]:
    """Run AutomatedSGIL once for this combo and return summary metrics."""

    import project_sgil.constants as constants

    # Patch constants (in-memory)
    constants.HEADING_ERROR_DEG = int(combo.heading_error_deg)
    constants.HEADING_SWEEP_RANGE_DEG = float(combo.heading_sweep_range_deg)
    constants.MAX_ESTIMATE_DIST_FROM_SNAPPED_POSITION_SWEEP = int(
        combo.max_estimate_dist_from_snapped_position_sweep
    )
    constants.AOI_RADIUS_M = int(combo.aoi_radius_m)

    # AOI angle depends on HEADING_ERROR_DEG and H_FOV_DEG
    constants.AOI_ANGLE_DEG = (
        float(constants.H_FOV_DEG) + float(constants.HEADING_ERROR_DEG)
    ) / 2.0

    # Reload modules that import constants at import-time
    import project_sgil.localization.tree_matcher as tree_matcher
    import project_sgil.automated_sgil as automated_sgil

    importlib.reload(tree_matcher)
    importlib.reload(automated_sgil)

    # Run
    runner = automated_sgil.AutomatedSGIL()
    results = runner.run()

    # Compute mean over finite SGIL errors
    sgil_errs = [getattr(r, "sgil_err_m", float("nan")) for r in results]
    mean_err = _mean_finite(sgil_errs)

    return {
        "heading_error_deg": combo.heading_error_deg,
        "heading_sweep_range_deg": combo.heading_sweep_range_deg,
        "max_estimate_dist_from_snapped_position_sweep": (
            combo.max_estimate_dist_from_snapped_position_sweep
        ),
        "aoi_radius_m": combo.aoi_radius_m,
        "mean_sgil_err_m": mean_err,
        "n_results": len(results),
    }


def main() -> None:
    combos = [
        Combo(
            heading_error_deg=h_err,
            heading_sweep_range_deg=h_range,
            max_estimate_dist_from_snapped_position_sweep=max_d,
            aoi_radius_m=aoi_r,
        )
        for h_err, h_range, max_d, aoi_r in itertools.product(
            range(2, 6),
            _frange(0.5, 2.5, 0.5),
            range(3, 6),
            [75]#range(65, 81, 5),
        )
    ]

    out_csv = "grid_search_automated_sgil_configs_results.csv"
    fieldnames = [
        "heading_error_deg",
        "heading_sweep_range_deg",
        "max_estimate_dist_from_snapped_position_sweep",
        "aoi_radius_m",
        "mean_sgil_err_m",
        "n_results",
        "runtime_s",
    ]

    best_row: dict[str, float | int | str | None] | None = None

    # Use newline='' for Windows CSV correctness
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos, start=1):
            t0 = time.time()
            print(
                f"\n[{i}/{len(combos)}] "
                f"HEADING_ERROR_DEG={combo.heading_error_deg} | "
                f"HEADING_SWEEP_RANGE_DEG={combo.heading_sweep_range_deg} | "
                f"MAX_ESTIMATE_DIST_FROM_SNAPPED_POSITION_SWEEP="
                f"{combo.max_estimate_dist_from_snapped_position_sweep} | "
                f"AOI_RADIUS_M={combo.aoi_radius_m}"
            )

            try:
                row = _evaluate_combo(combo)
            except Exception as e:
                row = {
                    "heading_error_deg": combo.heading_error_deg,
                    "heading_sweep_range_deg": combo.heading_sweep_range_deg,
                    "max_estimate_dist_from_snapped_position_sweep": (
                        combo.max_estimate_dist_from_snapped_position_sweep
                    ),
                    "aoi_radius_m": combo.aoi_radius_m,
                    "mean_sgil_err_m": None,
                    "n_results": 0,
                    "error": repr(e),
                }
                print(f"  -> FAILED: {e!r}")

            runtime_s = time.time() - t0
            mean_err = row.get("mean_sgil_err_m", None)
            print(f"  -> mean_sgil_err_m={mean_err} | runtime_s={runtime_s:.1f}")

            writer.writerow(
                {
                    "heading_error_deg": row.get("heading_error_deg"),
                    "heading_sweep_range_deg": row.get("heading_sweep_range_deg"),
                    "max_estimate_dist_from_snapped_position_sweep": row.get(
                        "max_estimate_dist_from_snapped_position_sweep"
                    ),
                    "aoi_radius_m": row.get("aoi_radius_m"),
                    "mean_sgil_err_m": mean_err,
                    "n_results": row.get("n_results"),
                    "runtime_s": round(runtime_s, 3),
                }
            )
            f.flush()

            if mean_err is None:
                continue
            if (
                best_row is None
                or (best_row.get("mean_sgil_err_m") is None)
                or float(mean_err)
                < float(best_row["mean_sgil_err_m"])  # type: ignore[index]
            ):
                best_row = dict(row)

    print("\n=== Done ===")
    if best_row is None:
        print("No successful runs produced a finite mean SGIL error.")
        return

    print("Best configuration:")
    print(
        "  "
        + " | ".join(
            [
                f"HEADING_ERROR_DEG={best_row['heading_error_deg']}",
                f"HEADING_SWEEP_RANGE_DEG={best_row['heading_sweep_range_deg']}",
                f"MAX_ESTIMATE_DIST_FROM_SNAPPED_POSITION_SWEEP="
                f"{best_row['max_estimate_dist_from_snapped_position_sweep']}",
                f"AOI_RADIUS_M={best_row['aoi_radius_m']}",
                f"mean_sgil_err_m={best_row['mean_sgil_err_m']}",
            ]
        )
    )
    print(f"\nSaved all results to: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
