"""Snakemake driver for the lab meeting TeLC bilateral angle stack panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    _repo_root = Path(__file__).resolve().parents[6]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import matplotlib

matplotlib.use("Agg")

from workflow.figures.lab_meeting_100926.telc_angle.figure_options_loader import (
    FIGURE_OPTIONS_YAML,
    get_bilateral_stack_panel_options,
)
from workflow.figures.lab_meeting_100926.telc_angle.panels.panel_bilateral_stack.load_data import (
    get_bilateral_stack_data,
)

from figures.components.whisker_angle_stack.draw import save_whisker_angle_stack_pdf


def run_panel_bilateral_stack(
    output: Path,
    repo_root: Path,
    options_path: Path = FIGURE_OPTIONS_YAML,
) -> None:
    """Load data and write the bilateral whisker angle stack PDF."""
    panel_opts = get_bilateral_stack_panel_options(options_path)
    data = get_bilateral_stack_data(repo_root, options_path)
    save_whisker_angle_stack_pdf(
        output,
        data,
        panel_opts.layout,
        panel_opts.plot,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the bilateral stack panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--figure-options", type=Path, default=FIGURE_OPTIONS_YAML)
    args = parser.parse_args(argv)
    run_panel_bilateral_stack(
        args.output.resolve(),
        args.repo_root.resolve(),
        args.figure_options.resolve(),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
