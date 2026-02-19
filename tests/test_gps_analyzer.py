import math

from project_sgil.localization.gps_analyzer import GPSAnalyzer


def test_heading_east_is_around_90_deg() -> None:
    a = GPSAnalyzer(window_size=5, min_distance_m=0.0)

    # Move east along the equator
    a.update(0.0, 0.0, 0.0)
    a.update(1.0, 0.0, 0.0001)
    a.update(2.0, 0.0, 0.0002)

    h = a.averaged_heading_deg()
    assert h is not None
    assert 80.0 <= h <= 100.0


def test_wraparound_mean_near_zero() -> None:
    a = GPSAnalyzer(window_size=5, min_distance_m=0.0)

    # Create two segments with headings around 359 and 1 degrees.
    # Use tiny movements so bearings are well-defined.
    a.update(0.0, 0.0, 0.0)
    # Slightly west of north-west gives ~359-ish when going to near-north slightly west
    a.update(1.0, 0.001, -1e-6)
    # Slightly east to make small positive bearing on next segment
    a.update(2.0, 0.002, 1e-6)

    h = a.averaged_heading_deg()
    assert h is not None

    # Accept either side of 0 due to numeric noise (e.g., 359.8 or 0.2)
    assert (h <= 5.0) or (h >= 355.0)


def test_stationary_returns_none() -> None:
    a = GPSAnalyzer(window_size=5, min_distance_m=10.0)

    a.update(0.0, 1.0, 1.0)
    a.update(1.0, 1.0, 1.0)
    a.update(2.0, 1.0, 1.0)

    assert a.averaged_heading_deg() is None


def test_out_of_order_ignored() -> None:
    a = GPSAnalyzer(window_size=5, min_distance_m=0.0)

    a.update(10.0, 0.0, 0.0)
    a.update(9.0, 0.0, 0.0001)  # ignored

    assert a.averaged_heading_deg() is None
    assert len(a._fixes) == 1  # type: ignore[attr-defined]

