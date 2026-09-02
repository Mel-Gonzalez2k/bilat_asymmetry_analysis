"""Load lab meeting TeLC angle figure options from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

from figures.components.whisker_angle_stack.options import (
    WhiskerAngleStackLayoutOptions,
    WhiskerAngleStackPlotOptions,
)

FIGURE_OPTIONS_YAML = Path(
    "workflow/figures/lab_meeting_100926/telc_angle/figure_options.yaml"
)
_BILATERAL_STACK_PANEL = "bilateral_stack"


class TelcAngleTimeRange(BaseModel):
    """Inclusive time window for whisker angle plotting."""

    model_config = {"frozen": True}

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_range(self) -> TelcAngleTimeRange:
        if self.end < self.start:
            raise ValueError("time_range.end must be >= time_range.start")
        return self


class TelcAngleExampleOptions(BaseModel):
    """One TeLC session example for whisker angle panels."""

    model_config = {"frozen": True}

    mouse_id: str
    experiment_id: str
    time_range: TelcAngleTimeRange
    left_whisker_key: str | None = None
    right_whisker_key: str | None = None


class TelcAnglePanelOptions(BaseModel):
    """Layout and plot options for one whisker angle stack panel."""

    model_config = {"frozen": True}

    example: str
    layout: WhiskerAngleStackLayoutOptions = Field(
        default_factory=WhiskerAngleStackLayoutOptions
    )
    plot: WhiskerAngleStackPlotOptions = Field(
        default_factory=WhiskerAngleStackPlotOptions
    )


class TelcAngleOptions(BaseModel):
    """Top-level lab meeting TeLC angle figure options."""

    model_config = {"frozen": True}

    examples: dict[str, TelcAngleExampleOptions] = Field(default_factory=dict)
    panels: dict[str, TelcAnglePanelOptions] = Field(default_factory=dict)


def get_telc_angle_options(
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> TelcAngleOptions:
    """Parse TeLC angle figure YAML into typed options."""
    yaml_conf = OmegaConf.load(options_path)
    if "telc_angle" not in yaml_conf:
        msg = f"Missing telc_angle section in {options_path}"
        raise ValueError(msg)
    container = cast(
        dict[str, Any],
        OmegaConf.to_container(yaml_conf.telc_angle, resolve=True),
    )
    return TelcAngleOptions(**container)


def get_bilateral_stack_panel_options(
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> TelcAnglePanelOptions:
    """Return panel options for the bilateral stack panel."""
    opts = get_telc_angle_options(options_path)
    panel = opts.panels.get(_BILATERAL_STACK_PANEL)
    if panel is None:
        msg = (
            f"Missing telc_angle.panels.{_BILATERAL_STACK_PANEL} "
            f"in {options_path}"
        )
        raise ValueError(msg)
    return panel


def get_telc_angle_example(
    example_key: str,
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> TelcAngleExampleOptions:
    """Return one named example from TeLC angle figure YAML."""
    opts = get_telc_angle_options(options_path)
    example = opts.examples.get(example_key)
    if example is None:
        msg = f"Unknown telc_angle.examples key: {example_key!r}"
        raise ValueError(msg)
    return example
