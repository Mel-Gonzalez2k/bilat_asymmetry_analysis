"""Render and run Neuralyzer for one TeLC left-whisker base-angle job."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.discover_whiskers import infer_whisker_side  # noqa: E402
from pipeline.neuralyzer_json import (  # noqa: E402
    NeuralyzerContext,
    render_pipeline,
    resolve_neuralyzer_runner,
    telc_output_dir,
    template_name_for_key,
    write_rendered_pipeline,
)

_REGISTRY_KEY = "telc_left_base_angle"


def run_telc_neuralyzer(
    *,
    repo_root: Path,
    mouse_id: str,
    experiment_id: str,
    whisker_key: str,
    whisker_csv: Path,
    output_dir: Path | None = None,
    registry_path: Path | None = None,
    template_dir: Path | None = None,
    neuralyzer_runner: str | Path | None = None,
    whisker_side: str | None = None,
    dry_run: bool = False,
    allow_placeholders: bool = False,
) -> list[Path]:
    """Render JSON and optionally invoke Neuralyzer for one whisker CSV.

    Parameters
    ----------
    repo_root
        Repository root.
    mouse_id
        Ledger mouse identifier.
    experiment_id
        Ledger experiment identifier.
    whisker_key
        Stable whisker key used in output paths and Neuralyzer data keys.
    whisker_csv
        Path to the whisker tracking CSV input.
    output_dir
        Directory for ``rendered_pipeline.json`` and ``base_angle.csv``.
    registry_path
        Neuralyzer registry YAML path.
    template_dir
        Directory containing Neuralyzer JSON templates.
    neuralyzer_runner
        Optional explicit path to ``DataManagerPipelineRunner.exe``.
    whisker_side
        Whisker side for angle-axis parameters (``left`` or ``right``).
        Inferred from ``whisker_csv`` when omitted.
    dry_run
        If true, write JSON and skip the Neuralyzer executable.
    allow_placeholders
        If true, allow registry transform names beginning with
        ``TODO Neuralyzer``.

    Returns
    -------
    list[Path]
        Expected Neuralyzer CSV output paths.
    """
    if registry_path is None:
        registry_path = repo_root / "pipeline" / "neuralyzer_registry.yaml"
    if template_dir is None:
        template_dir = repo_root / "pipeline" / "neuralyzer_templates"
    if output_dir is None:
        output_dir = telc_output_dir(
            repo_root,
            mouse_id,
            experiment_id,
            whisker_key,
        )

    if whisker_side is None:
        whisker_side = infer_whisker_side(whisker_csv)

    context = NeuralyzerContext(
        repo_root=repo_root,
        experiment="TeLC",
        animal=mouse_id,
        session=experiment_id,
        whisker=whisker_key,
        whisker_csv=whisker_csv,
        whisker_side=whisker_side,
        output_dir=output_dir,
    )
    template_name = template_name_for_key(registry_path, _REGISTRY_KEY)
    rendered = render_pipeline(
        template_dir / template_name,
        registry_path,
        _REGISTRY_KEY,
        context,
        allow_placeholders=allow_placeholders,
    )
    rendered_json = output_dir / "rendered_pipeline.json"
    expected = write_rendered_pipeline(rendered, rendered_json)
    print(f"Wrote Neuralyzer JSON: {rendered_json}")

    if dry_run:
        for path in expected:
            print(f"Expected output: {path}")
        return expected

    runner = resolve_neuralyzer_runner(neuralyzer_runner)
    subprocess.run([str(runner), str(rendered_json)], check=True)
    _check_outputs(expected)
    return expected


def _check_outputs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        msg = "Neuralyzer did not create expected outputs: " + ", ".join(
            str(path) for path in missing
        )
        raise FileNotFoundError(msg)


def main() -> None:
    """Parse command-line arguments and run one TeLC Neuralyzer job."""
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    run_telc_neuralyzer(
        repo_root=repo_root,
        mouse_id=args.mouse_id,
        experiment_id=args.experiment_id,
        whisker_key=args.whisker_key,
        whisker_csv=args.whisker_csv.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        registry_path=args.registry.resolve(),
        template_dir=args.template_dir.resolve(),
        neuralyzer_runner=args.neuralyzer_runner,
        whisker_side=args.whisker_side,
        dry_run=args.dry_run,
        allow_placeholders=args.allow_placeholders,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render and run Neuralyzer for one TeLC left-whisker base-angle job."
        )
    )
    parser.add_argument("--mouse-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--whisker-key", required=True)
    parser.add_argument("--whisker-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--registry",
        type=Path,
        default=_REPO_ROOT / "pipeline" / "neuralyzer_registry.yaml",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=_REPO_ROOT / "pipeline" / "neuralyzer_templates",
    )
    parser.add_argument("--neuralyzer-runner")
    parser.add_argument(
        "--whisker-side",
        choices=("left", "right"),
        help="Whisker side for angle axes (default: infer from whisker-csv path).",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow registry transform names beginning with TODO Neuralyzer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render JSON and print expected outputs without running Neuralyzer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
