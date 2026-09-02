"""Tests for TeLC Snakemake job loading."""

from __future__ import annotations

from pathlib import Path

from workflow.processing.scripts.telc_jobs import (
    jobs_to_expand_dict,
    load_telc_jobs,
    whisker_csv_lookup,
)


def test_load_telc_jobs_from_mock_session(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ledger = data_dir / "telc.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("Mouse_ID,Experiment_ID\nM1,Day1\n", encoding="utf-8")

    session = data_dir / "external" / "dynamic" / "TeLC" / "M1" / "Day1"
    whisker_csv = session / "whiskers" / "left" / "w1" / "track.csv"
    whisker_csv.parent.mkdir(parents=True)
    whisker_csv.write_text("Time,X,Y\n", encoding="utf-8")

    jobs = load_telc_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].mouse_id == "M1"
    assert jobs[0].experiment_id == "Day1"
    assert jobs[0].whisker_key == "left__w1__track"
    assert jobs[0].whisker_side == "left"
    assert jobs[0].whisker_csv == whisker_csv.resolve()


def test_jobs_to_expand_dict_and_lookup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ledger = data_dir / "telc.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("Mouse_ID,Experiment_ID\nM1,Day1\n", encoding="utf-8")

    session = data_dir / "external" / "dynamic" / "TeLC" / "M1" / "Day1"
    whisker_csv = session / "whiskers" / "left" / "track.csv"
    whisker_csv.parent.mkdir(parents=True)
    whisker_csv.write_text("Time,X,Y\n", encoding="utf-8")

    jobs = load_telc_jobs(tmp_path)
    expand_dict = jobs_to_expand_dict(jobs)
    assert expand_dict == {
        "mouse_id": ["M1"],
        "experiment_id": ["Day1"],
        "whisker_key": ["left__track"],
    }

    lookup = whisker_csv_lookup(jobs)
    assert lookup[("M1", "Day1", "left__track")] == whisker_csv.resolve()


def test_load_telc_jobs_both_sides(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ledger = data_dir / "telc.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("Mouse_ID,Experiment_ID\nM1,Day1\n", encoding="utf-8")

    session = data_dir / "external" / "dynamic" / "TeLC" / "M1" / "Day1"
    left_csv = session / "whiskers" / "left" / "track.csv"
    right_csv = session / "whiskers" / "right" / "track.csv"
    left_csv.parent.mkdir(parents=True)
    right_csv.parent.mkdir(parents=True)
    left_csv.write_text("Time,X,Y\n", encoding="utf-8")
    right_csv.write_text("Time,X,Y\n", encoding="utf-8")

    jobs = load_telc_jobs(tmp_path)
    assert len(jobs) == 2
    assert {job.whisker_key for job in jobs} == {"left__track", "right__track"}
    assert {job.whisker_side for job in jobs} == {"left", "right"}


def test_load_telc_jobs_empty_when_no_whiskers(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ledger = data_dir / "telc.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("Mouse_ID,Experiment_ID\nM1,Day1\n", encoding="utf-8")

    jobs = load_telc_jobs(tmp_path)
    assert jobs == []
    assert jobs_to_expand_dict(jobs) == {
        "mouse_id": [],
        "experiment_id": [],
        "whisker_key": [],
    }
