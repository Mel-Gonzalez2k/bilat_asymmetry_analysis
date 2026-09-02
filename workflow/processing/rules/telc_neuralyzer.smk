from pathlib import Path


rule neuralyzer_telc_base_angle:
    output:
        "data/processed/TeLC/{mouse_id}/{experiment_id}/{whisker_key}/base_angle.csv",
    params:
        whisker_csv=lambda wildcards: str(
            WHISKER_CSV_LOOKUP[
                (wildcards.mouse_id, wildcards.experiment_id, wildcards.whisker_key)
            ]
        ),
        output_dir=lambda wildcards, output: str(Path(output[0]).parent),
    shell:
        (
            "uv run python pipeline/run_telc_neuralyzer.py "
            "--mouse-id {wildcards.mouse_id} "
            "--experiment-id {wildcards.experiment_id} "
            "--whisker-key {wildcards.whisker_key} "
            "--whisker-csv {params.whisker_csv} "
            "--output-dir {params.output_dir}"
        )
