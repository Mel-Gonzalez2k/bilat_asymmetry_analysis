"""Tests for base_angle.csv loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from figures.loaders.base_angle_csv import (
    load_base_angle_csv,
    slice_base_angle_series,
)


def test_load_and_slice_base_angle_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "base_angle.csv"
    csv_path.write_text(
        "Time,Data\n0,1.0\n1,2.0\n2,3.0\n3,4.0\n",
        encoding="utf-8",
    )
    series = load_base_angle_csv(csv_path)
    assert np.allclose(series.time, [0, 1, 2, 3])
    assert np.allclose(series.angle, [1, 2, 3, 4])

    sliced = slice_base_angle_series(series, 1, 2)
    assert np.allclose(sliced.time, [1, 2])
    assert np.allclose(sliced.angle, [2, 3])


def test_slice_base_angle_series_empty_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "base_angle.csv"
    csv_path.write_text("Time,Data\n0,1.0\n", encoding="utf-8")
    series = load_base_angle_csv(csv_path)
    with pytest.raises(ValueError, match="No base-angle samples"):
        slice_base_angle_series(series, 5, 10)
