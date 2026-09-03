"""Tests for per-side whisker key resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow.processing.scripts.telc_jobs import resolve_whisker_keys_by_side


def test_resolve_whisker_keys_by_side(tmp_path: Path) -> None:
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

    keys = resolve_whisker_keys_by_side(tmp_path, "M1", "Day1")
    assert keys == {"left": "left__track", "right": "right__track"}


def test_resolve_whisker_keys_by_side_ambiguous_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ledger = data_dir / "telc.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("Mouse_ID,Experiment_ID\nM1,Day1\n", encoding="utf-8")

    session = data_dir / "external" / "dynamic" / "TeLC" / "M1" / "Day1"
    left_a = session / "whiskers" / "left" / "a.csv"
    left_b = session / "whiskers" / "left" / "b.csv"
    left_a.parent.mkdir(parents=True)
    left_a.write_text("Time,X,Y\n", encoding="utf-8")
    left_b.write_text("Time,X,Y\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple left whiskers"):
        resolve_whisker_keys_by_side(tmp_path, "M1", "Day1")
