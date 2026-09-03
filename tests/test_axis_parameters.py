"""Tests for whisker-side axis parameters."""

from __future__ import annotations

import pytest

from pipeline.discover_whiskers import infer_whisker_side
from pipeline.neuralyzer_json import axis_parameters_from_side


def test_axis_parameters_left() -> None:
    assert axis_parameters_from_side("left") == {
        "axis_x_x": -1.0,
        "axis_x_y": 0.0,
        "axis_y_x": 0.0,
        "axis_y_y": 1.0,
    }


def test_axis_parameters_right() -> None:
    assert axis_parameters_from_side("right") == {
        "axis_x_x": 1.0,
        "axis_x_y": 0.0,
        "axis_y_x": 0.0,
        "axis_y_y": 1.0,
    }


def test_axis_parameters_invalid_side() -> None:
    with pytest.raises(ValueError, match="Unknown whisker side"):
        axis_parameters_from_side("top")


def test_infer_whisker_side(tmp_path) -> None:
    left_csv = tmp_path / "whiskers" / "left" / "a" / "trace.csv"
    left_csv.parent.mkdir(parents=True)
    left_csv.touch()
    right_csv = tmp_path / "whiskers" / "right" / "trace.csv"
    right_csv.parent.mkdir(parents=True)
    right_csv.touch()

    assert infer_whisker_side(left_csv) == "left"
    assert infer_whisker_side(right_csv) == "right"
