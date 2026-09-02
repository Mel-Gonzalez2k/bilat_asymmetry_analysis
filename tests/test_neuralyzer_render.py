# TeLC left-whisker base angle (Neuralyzer rendering)

from __future__ import annotations

from pathlib import Path

from pipeline.neuralyzer_json import (
    NeuralyzerContext,
    render_pipeline,
    write_rendered_pipeline,
)


def test_render_telc_base_angle_pipeline(tmp_path: Path) -> None:
    repo_root = tmp_path
    whisker_csv = tmp_path / "whiskers" / "left" / "trace.csv"
    whisker_csv.parent.mkdir(parents=True)
    whisker_csv.write_text("Time,X,Y\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    registry = (
        Path(__file__).resolve().parents[1] / "pipeline" / "neuralyzer_registry.yaml"
    )
    template_dir = Path(__file__).resolve().parents[1] / "pipeline" / "neuralyzer_templates"

    context = NeuralyzerContext(
        repo_root=repo_root,
        experiment="TeLC",
        animal="M1",
        session="Day1",
        whisker="trace",
        whisker_csv=whisker_csv,
        whisker_side="left",
        output_dir=output_dir,
    )
    rendered = render_pipeline(
        template_dir / "whisker_base_angle_template.json",
        registry,
        "telc_left_base_angle",
        context,
    )
    expected = write_rendered_pipeline(rendered, output_dir / "rendered_pipeline.json")

    assert len(rendered["data"]) == 1
    assert rendered["data"][0]["filepath"] == str(whisker_csv.resolve()).replace(
        "\\", "/"
    )
    assert len(rendered["transformations"]["steps"]) == 1
    assert rendered["transformations"]["steps"][0]["transform_name"] == (
        "Calculate Line Angle"
    )
    params = rendered["transformations"]["steps"][0]["parameters"]
    assert params["axis_x_x"] == -1.0
    assert params["axis_x_y"] == 0.0
    assert params["axis_y_x"] == 0.0
    assert params["axis_y_y"] == 1.0
    assert expected == [output_dir / "base_angle.csv"]
