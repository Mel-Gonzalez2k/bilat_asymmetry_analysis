"""Render checked-in Neuralyzer JSON templates for Snakemake pipeline stages."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")
_PLACEHOLDER_PREFIX = "TODO Neuralyzer"
_SINC_UPSAMPLE_PIPELINE = (
    "${repo_root}/pipeline/neuralyzer_templates/pipelines/sinc_upsample.json"
)


@dataclass(frozen=True)
class NeuralyzerContext:
    """Session metadata used to render a Neuralyzer JSON template.

    Parameters
    ----------
    repo_root
        Repository root. Stored in JSON as ``repo_root``.
    experiment
        Experiment folder name, for example ``TeLC``.
    animal
        Animal identifier from the ledger.
    session
        Session identifier from the ledger.
    whisker
        Whisker identifier used in Neuralyzer data keys.
    whisker_csv
        Absolute path to the whisker tracking CSV input file.
    whisker_side
        Whisker side for angle-axis parameters: ``left`` or ``right``.
    output_dir
        Directory where rendered JSON and Neuralyzer CSV outputs are written.
    """

    repo_root: Path
    experiment: str
    animal: str
    session: str
    whisker: str
    whisker_csv: Path
    whisker_side: str
    output_dir: Path


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load the Neuralyzer registry YAML file.

    Parameters
    ----------
    path
        Registry YAML path.

    Returns
    -------
    dict[str, Any]
        Parsed registry mapping.

    Raises
    ------
    ValueError
        If the file does not contain a mapping.
    """
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    if not isinstance(registry, dict):
        msg = f"Neuralyzer registry must be a mapping: {registry_path}"
        raise ValueError(msg)
    return registry


def telc_output_dir(
    repo_root: str | Path,
    mouse_id: str,
    experiment_id: str,
    whisker_key: str,
) -> Path:
    """Return the default processed-output directory for one TeLC whisker."""
    return (
        Path(repo_root)
        / "data"
        / "processed"
        / "TeLC"
        / mouse_id
        / experiment_id
        / whisker_key
    )


def axis_parameters_from_side(side: str) -> dict[str, float]:
    """Return Neuralyzer angle-axis parameters for a whisker side.

    Parameters
    ----------
    side
        Whisker side: ``left`` or ``right``.

    Returns
    -------
    dict[str, float]
        Values for ``axis_x_x``, ``axis_x_y``, ``axis_y_x``, ``axis_y_y``.

    Raises
    ------
    ValueError
        If ``side`` is not ``left`` or ``right``.
    """
    lookup = {
        "left": {
            "axis_x_x": -1.0,
            "axis_x_y": 0.0,
            "axis_y_x": 0.0,
            "axis_y_y": 1.0,
        },
        "right": {
            "axis_x_x": 1.0,
            "axis_x_y": 0.0,
            "axis_y_x": 0.0,
            "axis_y_y": 1.0,
        },
    }
    key = side.strip().lower()
    if key not in lookup:
        msg = f"Unknown whisker side: {side!r} (expected 'left' or 'right')"
        raise ValueError(msg)
    return lookup[key]


def axis_parameters_from_facing(facing: str) -> dict[str, float]:
    """Return Neuralyzer angle-axis parameters from ``Animal Facing``.

    Parameters
    ----------
    facing
        Ledger screen direction: ``Top``, ``Bottom``, ``Left``, or ``Right``.

    Returns
    -------
    dict[str, float]
        Values for ``axis_x_x``, ``axis_x_y``, ``axis_y_x``, ``axis_y_y``.

    Raises
    ------
    ValueError
        If ``facing`` is not recognized.

    Notes
    -----
    Image coordinates use x right and y down. The local x-axis follows the
    animal-facing vector. The local y-axis is perpendicular to x and gives a
    positive-determinant 2D basis in image coordinates.
    """
    lookup = {
        "Top": (0.0, -1.0),
        "Bottom": (0.0, 1.0),
        "Left": (-1.0, 0.0),
        "Right": (1.0, 0.0),
    }
    key = facing.strip().title()
    if key not in lookup:
        msg = f"Unknown Animal Facing: {facing!r}"
        raise ValueError(msg)
    axis_x_x, axis_x_y = lookup[key]
    axis_y_x = -axis_x_y
    axis_y_y = axis_x_x
    return {
        "axis_x_x": axis_x_x,
        "axis_x_y": axis_x_y,
        "axis_y_x": axis_y_x,
        "axis_y_y": axis_y_y,
    }


def template_name_for_key(registry_path: Path, registry_key: str) -> str:
    """Return the template filename declared for a registry key."""
    registry = load_registry(registry_path)
    key_spec = registry.get(registry_key)
    if not isinstance(key_spec, dict) or "template" not in key_spec:
        msg = f"Registry key {registry_key!r} must define a template"
        raise ValueError(msg)
    return str(key_spec["template"])


def render_pipeline(
    template_path: str | Path,
    registry_path: str | Path,
    registry_key: str,
    context: NeuralyzerContext,
    *,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    """Render a Neuralyzer JSON pipeline from a template and registry key.

    Parameters
    ----------
    template_path
        Checked-in Neuralyzer JSON template path.
    registry_path
        YAML registry path.
    registry_key
        Top-level registry key to render.
    context
        Session and output metadata.
    allow_placeholders
        If false, reject transform names beginning with ``TODO Neuralyzer``.

    Returns
    -------
    dict[str, Any]
        Rendered JSON object ready to write and pass to Neuralyzer.
    """
    template_file = Path(template_path)
    with template_file.open("r", encoding="utf-8") as stream:
        template = json.load(stream)

    registry = load_registry(registry_path)
    if registry_key not in registry:
        msg = f"Unknown Neuralyzer registry key: {registry_key!r}"
        raise ValueError(msg)
    spec = registry[registry_key]
    if not isinstance(spec, dict):
        msg = f"Registry key {registry_key!r} must contain a mapping"
        raise ValueError(msg)

    rendered = copy.deepcopy(template)
    variables = _variables_for_context(context)
    rendered["variables"] = {
        **rendered.get("variables", {}),
        **variables,
    }

    template_saves = list(rendered.get("saves") or [])
    template_commands = list(rendered.get("commands") or [])
    dynamic_values = axis_parameters_from_side(context.whisker_side)
    steps, saves = _steps_and_saves_from_spec(
        spec,
        dynamic_values,
        allow_placeholders=allow_placeholders,
    )
    rendered.setdefault("transformations", {})
    rendered["transformations"]["steps"] = steps

    upsample_factor = _upsample_factor_from_spec(spec, rendered)
    commands, upsample_saves = _commands_with_upsampling(
        template_commands,
        spec,
        template_saves,
        upsample_factor=upsample_factor,
    )
    rendered["commands"] = commands
    rendered["saves"] = saves + template_saves + upsample_saves
    return _expand_variables(rendered, rendered["variables"])


def write_rendered_pipeline(
    rendered: dict[str, Any],
    path: str | Path,
) -> list[Path]:
    """Write rendered JSON and return expected CSV output paths.

    Parameters
    ----------
    rendered
        Rendered JSON object from :func:`render_pipeline`.
    path
        Destination JSON path.

    Returns
    -------
    list[Path]
        Save paths declared by the rendered JSON.
    """
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rendered, stream, indent=4)
        stream.write("\n")
    return expected_output_paths(rendered)


def expected_output_paths(rendered: dict[str, Any]) -> list[Path]:
    """Return output paths declared by ``saves`` in rendered JSON."""
    saves = rendered.get("saves", [])
    if not isinstance(saves, list):
        msg = "Rendered Neuralyzer JSON field 'saves' must be a list"
        raise ValueError(msg)
    paths: list[Path] = []
    for save in saves:
        if not isinstance(save, dict) or "path" not in save:
            msg = "Every Neuralyzer save entry must contain a path"
            raise ValueError(msg)
        paths.append(Path(str(save["path"])))
    return paths


def resolve_neuralyzer_runner(
    runner: str | Path | None = None,
) -> Path:
    """Resolve the Neuralyzer CLI executable path.

    Resolution order is explicit ``runner``, ``NEURALYZER_RUNNER``, then
    ``DataManagerPipelineRunner.exe`` on ``PATH``.
    """
    candidates: list[str | Path] = []
    if runner is not None:
        candidates.append(runner)
    env_runner = os.environ.get("NEURALYZER_RUNNER")
    if env_runner:
        candidates.append(env_runner)

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
        msg = f"Neuralyzer runner does not exist: {path}"
        raise FileNotFoundError(msg)

    from_path = shutil.which("DataManagerPipelineRunner.exe")
    if from_path is not None:
        return Path(from_path)
    msg = (
        "Could not find DataManagerPipelineRunner.exe. Set "
        "NEURALYZER_RUNNER or pass --neuralyzer-runner."
    )
    raise FileNotFoundError(msg)


def _variables_for_context(context: NeuralyzerContext) -> dict[str, str]:
    return {
        "repo_root": context.repo_root.resolve().as_posix(),
        "experiment_type": context.experiment,
        "animal": context.animal,
        "session": context.session,
        "whisker_id": context.whisker,
        "whisker_csv_path": context.whisker_csv.resolve().as_posix(),
        "neuralyzer_output_dir": context.output_dir.resolve().as_posix(),
    }


def _steps_and_saves_from_spec(
    spec: dict[str, Any],
    dynamic_values: dict[str, float],
    *,
    allow_placeholders: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs = spec.get("outputs")
    if not isinstance(outputs, dict):
        msg = "Neuralyzer registry key must contain an 'outputs' mapping"
        raise ValueError(msg)

    steps: list[dict[str, Any]] = []
    saves: list[dict[str, Any]] = []
    for output_name, output_spec in outputs.items():
        if not isinstance(output_spec, dict):
            msg = f"Output {output_name!r} must contain a mapping"
            raise ValueError(msg)
        transform_name = str(output_spec["transform_name"])
        if (
            transform_name.startswith(_PLACEHOLDER_PREFIX)
            and not allow_placeholders
        ):
            msg = (
                f"Output {output_name!r} uses placeholder transform "
                f"{transform_name!r}"
            )
            raise ValueError(msg)

        parameters = dict(output_spec.get("parameters") or {})
        for name in output_spec.get("dynamic_parameters") or []:
            if name not in dynamic_values:
                msg = f"Unknown dynamic Neuralyzer parameter: {name!r}"
                raise ValueError(msg)
            parameters[name] = dynamic_values[name]

        output_key = output_spec["output_key"]
        steps.append(
            {
                "step_id": output_spec["step_id"],
                "transform_name": transform_name,
                "phase": int(output_spec.get("phase", 0)),
                "input_key": output_spec["input_key"],
                "output_key": output_key,
                "parameters": parameters,
            }
        )
        saves.append(_save_entry(output_key, output_spec))
    return steps, saves


def _save_entry(
    output_key: str,
    output_spec: dict[str, Any],
) -> dict[str, Any]:
    save_spec = output_spec.get("save")
    if not isinstance(save_spec, dict):
        msg = f"Output {output_key!r} must contain a save mapping"
        raise ValueError(msg)
    filename = str(save_spec["filename"])
    return {
        "data_key": output_key,
        "format": str(save_spec.get("format", "csv")),
        "path": "${neuralyzer_output_dir}/" + filename,
        "format_options": {
            "save_header": bool(save_spec.get("save_header", True)),
            "header": str(save_spec.get("header", "Time,Data")),
            "precision": int(save_spec.get("precision", 2)),
        },
    }


def _upsample_factor_from_spec(
    spec: dict[str, Any],
    template: dict[str, Any],
) -> int:
    """Return the kinematic upsampling factor declared in registry or template."""
    factor = spec.get("upsample_factor")
    if factor is None:
        template_vars = template.get("variables") or {}
        factor = template_vars.get("kinematic_upsample_factor")
    if factor is None:
        return 1
    parsed = int(factor)
    if parsed < 1:
        msg = f"upsample_factor must be >= 1, got {parsed}"
        raise ValueError(msg)
    return parsed


def _output_should_upsample(
    upsample_flag: object,
    *,
    upsample_factor: int,
) -> bool:
    if upsample_factor <= 1:
        return False
    if upsample_flag is None:
        return True
    return bool(upsample_flag)


def _upsampled_time_key(upsample_factor: int) -> str:
    return f"time_{upsample_factor}x"


def _upsampled_output_key(output_key: str, upsample_factor: int) -> str:
    return f"{output_key}_{upsample_factor}x"


def _upsampled_filename(filename: str, upsample_factor: int) -> str:
    path = Path(filename)
    return f"{path.stem}_{upsample_factor}x{path.suffix}"


def _upsample_timeframe_command(upsample_factor: int) -> dict[str, Any]:
    return {
        "command_name": "UpsampleTimeFrame",
        "parameters": {
            "source_time_key": "time",
            "output_time_key": _upsampled_time_key(upsample_factor),
            "upsampling_factor": upsample_factor,
            "overwrite": True,
        },
    }


def _sinc_upsample_command(
    input_key: str,
    output_key: str,
    *,
    upsample_factor: int,
) -> dict[str, Any]:
    return {
        "command_name": "RunTransformsV2Pipeline",
        "parameters": {
            "input_key": input_key,
            "output_key": output_key,
            "output_time_key": _upsampled_time_key(upsample_factor),
            "pipeline_path": _SINC_UPSAMPLE_PIPELINE,
        },
    }


def _upsample_save_from_native(
    output_key: str,
    save_spec: dict[str, Any],
    *,
    upsample_factor: int,
) -> dict[str, Any]:
    filename = _upsampled_filename(str(save_spec["filename"]), upsample_factor)
    upsampled_key = _upsampled_output_key(output_key, upsample_factor)
    return {
        "data_key": upsampled_key,
        "format": str(save_spec.get("format", "csv")),
        "path": "${neuralyzer_output_dir}/" + filename,
        "format_options": {
            "save_header": bool(save_spec.get("save_header", True)),
            "header": str(save_spec.get("header", "Time,Data")),
            "precision": int(save_spec.get("precision", 2)),
        },
    }


def _commands_with_upsampling(
    template_commands: list[dict[str, Any]],
    spec: dict[str, Any],
    template_saves: list[dict[str, Any]],
    *,
    upsample_factor: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble command list and upsampled save entries for rendered JSON."""
    commands: list[dict[str, Any]] = []
    upsample_saves: list[dict[str, Any]] = []
    if upsample_factor > 1:
        commands.append(_upsample_timeframe_command(upsample_factor))
    commands.extend(template_commands)

    outputs = spec.get("outputs")
    if isinstance(outputs, dict):
        for output_spec in outputs.values():
            if not isinstance(output_spec, dict):
                continue
            if not _output_should_upsample(
                output_spec.get("upsample"),
                upsample_factor=upsample_factor,
            ):
                continue
            input_key = str(output_spec["output_key"])
            output_key = _upsampled_output_key(input_key, upsample_factor)
            commands.append(
                _sinc_upsample_command(
                    input_key,
                    output_key,
                    upsample_factor=upsample_factor,
                )
            )
            save_spec = output_spec.get("save")
            if isinstance(save_spec, dict):
                upsample_saves.append(
                    _upsample_save_from_native(
                        input_key,
                        save_spec,
                        upsample_factor=upsample_factor,
                    )
                )

    for save in template_saves:
        if not isinstance(save, dict):
            continue
        if not _output_should_upsample(
            save.get("upsample"),
            upsample_factor=upsample_factor,
        ):
            continue
        input_key = str(save["data_key"])
        output_key = _upsampled_output_key(input_key, upsample_factor)
        commands.append(
            _sinc_upsample_command(
                input_key,
                output_key,
                upsample_factor=upsample_factor,
            )
        )
        format_options = save.get("format_options")
        if not isinstance(format_options, dict):
            msg = (
                f"Template save for {input_key!r} must include format_options"
            )
            raise ValueError(msg)
        header = str(format_options.get("header", "Time,Data"))
        upsample_saves.append(
            _upsample_save_from_native(
                input_key,
                {
                    "filename": Path(str(save["path"])).name,
                    "header": header,
                    "precision": int(format_options.get("precision", 2)),
                    "save_header": bool(
                        format_options.get("save_header", True)
                    ),
                    "format": str(save.get("format", "csv")),
                },
                upsample_factor=upsample_factor,
            )
        )

    return commands, upsample_saves


def _expand_variables(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _VARIABLE_PATTERN.sub(
            lambda match: str(variables.get(match.group(1), match.group(0))),
            value,
        )
    if isinstance(value, list):
        return [_expand_variables(item, variables) for item in value]
    if isinstance(value, dict):
        return {
            key: _expand_variables(item, variables)
            for key, item in value.items()
        }
    return value
