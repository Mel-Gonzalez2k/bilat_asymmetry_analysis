"""Load bilateral whisker angle stack panel data."""

from __future__ import annotations

from pathlib import Path

from figures.components.whisker_angle_stack.draw import WhiskerAngleStackData
from figures.loaders.base_angle_csv import (
    load_base_angle_csv,
    slice_base_angle_series,
)
from workflow.figures.lab_meeting_100926.telc_angle.figure_options_loader import (
    FIGURE_OPTIONS_YAML,
    get_bilateral_stack_panel_options,
    get_telc_angle_example,
)
from workflow.processing.scripts.telc_jobs import (
    processed_base_angle_path,
    resolve_whisker_keys_by_side,
)

__all__ = ["get_bilateral_stack_data"]


def _resolve_whisker_keys(
    repo_root: Path,
    example_key: str,
    options_path: Path,
) -> dict[str, str]:
    example = get_telc_angle_example(example_key, options_path)
    defaults = resolve_whisker_keys_by_side(
        repo_root,
        example.mouse_id,
        example.experiment_id,
    )
    return {
        "left": example.left_whisker_key or defaults["left"],
        "right": example.right_whisker_key or defaults["right"],
    }


def get_bilateral_stack_data(
    repo_root: Path,
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> WhiskerAngleStackData:
    """Load sliced left/right base-angle series for the bilateral stack panel."""
    panel = get_bilateral_stack_panel_options(options_path)
    example = get_telc_angle_example(panel.example, options_path)
    keys = _resolve_whisker_keys(repo_root, panel.example, options_path)

    left_path = processed_base_angle_path(
        repo_root,
        example.mouse_id,
        example.experiment_id,
        keys["left"],
    )
    right_path = processed_base_angle_path(
        repo_root,
        example.mouse_id,
        example.experiment_id,
        keys["right"],
    )

    time_range = example.time_range
    left = slice_base_angle_series(
        load_base_angle_csv(left_path),
        time_range.start,
        time_range.end,
    )
    right = slice_base_angle_series(
        load_base_angle_csv(right_path),
        time_range.start,
        time_range.end,
    )
    return WhiskerAngleStackData(
        left_time=left.time,
        left_angle=left.angle,
        right_time=right.time,
        right_angle=right.angle,
    )
