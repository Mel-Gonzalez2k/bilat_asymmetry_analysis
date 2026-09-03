"""Tests for lab meeting TeLC angle figure YAML options."""

from __future__ import annotations

from pathlib import Path

from workflow.figures.lab_meeting_100926.telc_angle.figure_options_loader import (
    get_bilateral_stack_panel_options,
    get_telc_angle_example,
    get_telc_angle_options,
)


def test_telc_angle_options_from_repo_yaml() -> None:
    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "workflow/figures/lab_meeting_100926/telc_angle/figure_options.yaml"
    )
    opts = get_telc_angle_options(yaml_path)
    example = opts.examples["animal_1_day7"]
    assert example.mouse_id == "Animal_1"
    assert example.experiment_id == "Day7"
    assert example.time_range.start == 60000
    assert example.time_range.end == 62000

    panel = get_bilateral_stack_panel_options(yaml_path)
    assert panel.example == "animal_1_day7"
    assert panel.plot.ylabel == "Angle (rad)"
    assert panel.layout.figsize == (10.0, 4.0)


def test_get_telc_angle_example(tmp_path: Path) -> None:
    yaml_path = tmp_path / "figure_options.yaml"
    yaml_path.write_text(
        """
telc_angle:
  examples:
    ex1:
      mouse_id: M1
      experiment_id: Day1
      time_range:
        start: 1
        end: 3
  panels:
    bilateral_stack:
      example: ex1
""",
        encoding="utf-8",
    )
    example = get_telc_angle_example("ex1", yaml_path)
    assert example.mouse_id == "M1"
