from pathlib import Path

from workflow.figures.lab_meeting_100926.telc_angle.resolve_inputs import (
    FIGURE_OPTIONS_YAML,
    bilateral_stack_input_paths,
)

_BILATERAL_STACK_INPUTS = bilateral_stack_input_paths(Path(workflow.basedir).resolve())

LAB_MEETING_100926_TELC_ANGLE_PDF = (
    "results/figures/lab_meeting_100926/telc_angle/bilateral_stack.pdf"
)

include: "panels/panel_bilateral_stack/panel.smk"


rule lab_meeting_100926_telc_angle:
    input:
        LAB_MEETING_100926_TELC_ANGLE_PDF,
