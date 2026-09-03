"""Load Neuralyzer ``base_angle.csv`` time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_TIME_COLUMN = "Time"
_ANGLE_COLUMN = "Data"


@dataclass(frozen=True)
class BaseAngleSeries:
    """Whisker base-angle samples from a Neuralyzer CSV export."""

    time: np.ndarray
    angle: np.ndarray


def load_base_angle_csv(path: str | Path) -> BaseAngleSeries:
    """Load a ``base_angle.csv`` file produced by Neuralyzer.

    Parameters
    ----------
    path
        CSV path with ``Time`` and ``Data`` columns.

    Returns
    -------
    BaseAngleSeries
        Time and angle arrays (angle is column 2 / ``Data``).
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    if _TIME_COLUMN not in frame.columns or _ANGLE_COLUMN not in frame.columns:
        msg = (
            f"Expected columns {_TIME_COLUMN!r} and {_ANGLE_COLUMN!r} in "
            f"{csv_path}"
        )
        raise ValueError(msg)
    time = frame[_TIME_COLUMN].to_numpy(dtype=np.float64)
    angle = frame[_ANGLE_COLUMN].to_numpy(dtype=np.float64)
    return BaseAngleSeries(time=time, angle=angle)


def slice_base_angle_series(
    series: BaseAngleSeries,
    start: int,
    end: int,
) -> BaseAngleSeries:
    """Return samples with inclusive ``start <= time <= end``.

    Parameters
    ----------
    series
        Full base-angle series.
    start
        Inclusive start time (frame index).
    end
        Inclusive end time (frame index).

    Returns
    -------
    BaseAngleSeries
        Sliced series.

    Raises
    ------
    ValueError
        If ``end < start`` or the slice is empty.
    """
    if end < start:
        msg = f"time_range end ({end}) must be >= start ({start})"
        raise ValueError(msg)
    mask = (series.time >= start) & (series.time <= end)
    if not np.any(mask):
        msg = (
            f"No base-angle samples in inclusive time range [{start}, {end}]"
        )
        raise ValueError(msg)
    return BaseAngleSeries(
        time=series.time[mask],
        angle=series.angle[mask],
    )
