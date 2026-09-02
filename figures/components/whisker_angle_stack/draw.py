"""Draw stacked left/right whisker base-angle traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from figures.components.whisker_angle_stack.options import (
    WhiskerAngleStackLayoutOptions,
    WhiskerAngleStackPlotOptions,
)

__all__ = [
    "WhiskerAngleStackData",
    "draw_whisker_angle_stack",
    "save_whisker_angle_stack_pdf",
]


@dataclass(frozen=True)
class WhiskerAngleStackData:
    """Loaded angle traces for a bilateral whisker stack panel."""

    left_time: np.ndarray
    left_angle: np.ndarray
    right_time: np.ndarray
    right_angle: np.ndarray


def draw_whisker_angle_stack(
    axes: tuple[Axes, Axes],
    data: WhiskerAngleStackData,
    plot_opts: WhiskerAngleStackPlotOptions,
) -> None:
    """Draw left (top) and right (bottom) whisker angle traces."""
    left_ax, right_ax = axes
    left_ax.plot(
        data.left_time,
        data.left_angle,
        color="black",
        linewidth=plot_opts.linewidth,
    )
    right_ax.plot(
        data.right_time,
        data.right_angle,
        color="black",
        linewidth=plot_opts.linewidth,
    )

    left_ax.set_title(plot_opts.left_title)
    right_ax.set_title(plot_opts.right_title)
    right_ax.set_xlabel(plot_opts.xlabel)
    left_ax.set_ylabel(plot_opts.ylabel)
    right_ax.set_ylabel(plot_opts.ylabel)

    right_ax.set_xlim(
        min(data.left_time.min(), data.right_time.min()),
        max(data.left_time.max(), data.right_time.max()),
    )
    left_ax.tick_params(labelbottom=False)
    left_ax.spines["top"].set_visible(False)
    left_ax.spines["right"].set_visible(False)
    right_ax.spines["top"].set_visible(False)
    right_ax.spines["right"].set_visible(False)


def save_whisker_angle_stack_pdf(
    output: Path,
    data: WhiskerAngleStackData,
    layout_opts: WhiskerAngleStackLayoutOptions,
    plot_opts: WhiskerAngleStackPlotOptions,
) -> None:
    """Render and save a bilateral whisker angle stack PDF."""
    fig = Figure(figsize=layout_opts.figsize, layout="constrained")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=list(layout_opts.height_ratios),
        hspace=layout_opts.hspace,
    )
    left_ax = fig.add_subplot(grid[0, 0])
    right_ax = fig.add_subplot(grid[1, 0], sharex=left_ax)
    draw_whisker_angle_stack((left_ax, right_ax), data, plot_opts)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
