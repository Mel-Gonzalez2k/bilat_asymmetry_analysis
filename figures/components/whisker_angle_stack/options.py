"""Plot options for stacked left/right whisker angle panels."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WhiskerAngleStackPlotOptions(BaseModel):
    """Styling for bilateral whisker angle stack traces."""

    model_config = {"frozen": True}

    xlabel: str = "Time"
    ylabel: str = "Angle (rad)"
    left_title: str = "Left whisker"
    right_title: str = "Right whisker"
    linewidth: float = Field(default=0.8, gt=0.0)


class WhiskerAngleStackLayoutOptions(BaseModel):
    """Figure layout for a two-row whisker angle stack."""

    model_config = {"frozen": True}

    figsize: tuple[float, float] = (10.0, 4.0)
    height_ratios: tuple[float, float] = (1.0, 1.0)
    hspace: float = Field(default=0.08, ge=0.0)
