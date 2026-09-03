"""Build TeLC Snakemake job lists from the ledger and whisker discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.discover_whiskers import (
    discover_whisker_csvs,
    infer_whisker_side,
    whisker_key_from_path,
    whiskers_side_dir,
)

TELC_LEDGER_PATH = Path("data") / "telc.csv"
TELC_EXPERIMENT = "TeLC"
DYNAMIC_ROOT = Path("data") / "external" / "dynamic"
RAW_ROOT = Path("data") / "external" / "raw"


@dataclass(frozen=True)
class TelcJob:
    """One Neuralyzer job for a whisker CSV in a TeLC session."""

    mouse_id: str
    experiment_id: str
    whisker_key: str
    whisker_csv: Path
    whisker_side: str


def dynamic_session_dir(
    repo_root: str | Path,
    mouse_id: str,
    experiment_id: str,
) -> Path:
    """Return the dynamic external directory for one ledger row."""
    return (
        Path(repo_root)
        / DYNAMIC_ROOT
        / TELC_EXPERIMENT
        / mouse_id
        / experiment_id
    )


def raw_session_dir(
    repo_root: str | Path,
    mouse_id: str,
    experiment_id: str,
) -> Path:
    """Return the raw external directory for one ledger row."""
    return (
        Path(repo_root)
        / RAW_ROOT
        / TELC_EXPERIMENT
        / mouse_id
        / experiment_id
    )


def load_telc_jobs(repo_root: str | Path) -> list[TelcJob]:
    """Load TeLC jobs from the ledger and on-disk whisker CSV discovery.

    Parameters
    ----------
    repo_root
        Repository root containing ``data/telc.csv``.

    Returns
    -------
    list[TelcJob]
        One job per whisker CSV found under ``whiskers/left`` or
        ``whiskers/right`` for each ledger row. Rows with no whisker data
        contribute no jobs.
    """
    ledger_path = Path(repo_root) / TELC_LEDGER_PATH
    ledger = pd.read_csv(ledger_path)
    jobs: list[TelcJob] = []
    for _, row in ledger.iterrows():
        mouse_id = str(row["Mouse_ID"])
        experiment_id = str(row["Experiment_ID"])
        session_dir = dynamic_session_dir(repo_root, mouse_id, experiment_id)
        for csv_path, side in discover_whisker_csvs(session_dir):
            side_root = whiskers_side_dir(session_dir, side)
            jobs.append(
                TelcJob(
                    mouse_id=mouse_id,
                    experiment_id=experiment_id,
                    whisker_key=whisker_key_from_path(csv_path, side_root, side),
                    whisker_csv=csv_path,
                    whisker_side=infer_whisker_side(csv_path),
                )
            )
    return jobs


def jobs_to_expand_dict(jobs: list[TelcJob]) -> dict[str, list[str]]:
    """Convert jobs to keyword arguments for Snakemake ``expand()``."""
    if not jobs:
        return {"mouse_id": [], "experiment_id": [], "whisker_key": []}
    return {
        "mouse_id": [job.mouse_id for job in jobs],
        "experiment_id": [job.experiment_id for job in jobs],
        "whisker_key": [job.whisker_key for job in jobs],
    }


def whisker_csv_lookup(jobs: list[TelcJob]) -> dict[tuple[str, str, str], Path]:
    """Map ``(mouse_id, experiment_id, whisker_key)`` to whisker CSV paths."""
    return {
        (job.mouse_id, job.experiment_id, job.whisker_key): job.whisker_csv
        for job in jobs
    }


def processed_base_angle_path(
    repo_root: str | Path,
    mouse_id: str,
    experiment_id: str,
    whisker_key: str,
) -> Path:
    """Return the processed ``base_angle.csv`` path for one whisker job."""
    return (
        Path(repo_root)
        / "data"
        / "processed"
        / TELC_EXPERIMENT
        / mouse_id
        / experiment_id
        / whisker_key
        / "base_angle.csv"
    )


def resolve_whisker_keys_by_side(
    repo_root: str | Path,
    mouse_id: str,
    experiment_id: str,
) -> dict[str, str]:
    """Return the unique whisker key for each side in one session.

    Parameters
    ----------
    repo_root
        Repository root.
    mouse_id
        Ledger mouse identifier.
    experiment_id
        Ledger experiment identifier.

    Returns
    -------
    dict[str, str]
        Mapping ``left`` and ``right`` to whisker keys.

    Raises
    ------
    ValueError
        If a side has zero or multiple whisker keys.
    """
    jobs = load_telc_jobs(repo_root)
    session_jobs = [
        job
        for job in jobs
        if job.mouse_id == mouse_id and job.experiment_id == experiment_id
    ]
    by_side: dict[str, list[str]] = {"left": [], "right": []}
    for job in session_jobs:
        by_side[job.whisker_side].append(job.whisker_key)

    resolved: dict[str, str] = {}
    for side in ("left", "right"):
        keys = sorted(set(by_side[side]))
        if len(keys) == 0:
            msg = (
                f"No {side} whisker found for {mouse_id}/{experiment_id}. "
                "Add whisker tracking CSVs or set explicit whisker keys in YAML."
            )
            raise ValueError(msg)
        if len(keys) > 1:
            msg = (
                f"Multiple {side} whiskers found for {mouse_id}/{experiment_id}: "
                f"{keys}. Set explicit whisker keys in YAML."
            )
            raise ValueError(msg)
        resolved[side] = keys[0]
    return resolved
