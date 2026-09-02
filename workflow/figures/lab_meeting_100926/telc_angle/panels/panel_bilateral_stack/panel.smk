rule lab_meeting_100926_telc_angle_bilateral:
    input:
        script=(
            "workflow/figures/lab_meeting_100926/telc_angle/"
            "panels/panel_bilateral_stack/driver_panel_bilateral_stack.py"
        ),
        load_data=(
            "workflow/figures/lab_meeting_100926/telc_angle/"
            "panels/panel_bilateral_stack/load_data.py"
        ),
        left_base_angle=_BILATERAL_STACK_INPUTS["left_base_angle"],
        right_base_angle=_BILATERAL_STACK_INPUTS["right_base_angle"],
        figure_options=FIGURE_OPTIONS_YAML,
    output:
        LAB_MEETING_100926_TELC_ANGLE_PDF,
    shell:
        (
            "uv run python {input.script} "
            "--output {output} "
            "--repo-root . "
            "--figure-options {input.figure_options}"
        )
