from pathlib import Path
import sys

REPO_ROOT = Path(workflow.basedir).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.processing.scripts.telc_jobs import (  # noqa: E402
    jobs_to_expand_dict,
    load_telc_jobs,
    whisker_csv_lookup,
)

TELC_JOBS = load_telc_jobs(REPO_ROOT)
WHISKER_CSV_LOOKUP = whisker_csv_lookup(TELC_JOBS)


rule all:
    input:
        expand(
            "data/processed/TeLC/{mouse_id}/{experiment_id}/{whisker_key}/base_angle.csv",
            **jobs_to_expand_dict(TELC_JOBS),
        ),


include: "workflow/processing/rules/telc_neuralyzer.smk"
include: "workflow/figures/lab_meeting_100926/telc_angle/telc_angle.smk"
