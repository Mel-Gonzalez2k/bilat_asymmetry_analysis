"""Resolve Snakemake inputs for the lab meeting TeLC angle panel."""

from __future__ import annotations

from pathlib import Path

from workflow.figures.lab_meeting_100926.telc_angle.figure_options_loader import (
    FIGURE_OPTIONS_YAML,
    get_bilateral_stack_panel_options,
    get_telc_angle_example,
)
from workflow.processing.scripts.telc_jobs import (
    processed_base_angle_path,
    resolve_whisker_keys_by_side,
)

__all__ = [
    "FIGURE_OPTIONS_YAML",
    "bilateral_stack_input_paths",
]


def _resolve_example_whisker_keys(
    repo_root: Path,
    example_key: str,
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> dict[str, str]:
    """Return left/right whisker keys for one YAML example."""
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


def bilateral_stack_input_paths(
    repo_root: Path,
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> dict[str, Path]:
    """Return Snakemake input paths for the bilateral stack panel."""
    panel = get_bilateral_stack_panel_options(options_path)
    example = get_telc_angle_example(panel.example, options_path)
    keys = _resolve_example_whisker_keys(repo_root, panel.example, options_path)
    return {
        "left_base_angle": processed_base_angle_path(
            repo_root,
            example.mouse_id,
            example.experiment_id,
            keys["left"],
        ),
        "right_base_angle": processed_base_angle_path(
            repo_root,
            example.mouse_id,
            example.experiment_id,
            keys["right"],
        ),
    }
