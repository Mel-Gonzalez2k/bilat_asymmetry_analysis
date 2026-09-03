"""Tests for whisker CSV discovery."""

from __future__ import annotations

from pathlib import Path

from pipeline.discover_whiskers import (
    discover_left_whisker_csvs,
    discover_whisker_csvs,
    resolve_whisker_csv,
    whisker_key_from_path,
)


def test_discover_left_whisker_csvs_nested(tmp_path: Path) -> None:
    left = tmp_path / "whiskers" / "left"
    (left / "a").mkdir(parents=True)
    (left / "b").mkdir(parents=True)
    foo = left / "a" / "foo.csv"
    bar = left / "b" / "bar.csv"
    foo.write_text("Time,X,Y\n", encoding="utf-8")
    bar.write_text("Time,X,Y\n", encoding="utf-8")

    discovered = discover_left_whisker_csvs(tmp_path)
    assert discovered == sorted([foo.resolve(), bar.resolve()])


def test_discover_whisker_csvs_both_sides(tmp_path: Path) -> None:
    left_csv = tmp_path / "whiskers" / "left" / "a" / "foo.csv"
    right_csv = tmp_path / "whiskers" / "right" / "b" / "bar.csv"
    left_csv.parent.mkdir(parents=True)
    right_csv.parent.mkdir(parents=True)
    left_csv.write_text("Time,X,Y\n", encoding="utf-8")
    right_csv.write_text("Time,X,Y\n", encoding="utf-8")

    discovered = discover_whisker_csvs(tmp_path)
    assert discovered == [
        (left_csv.resolve(), "left"),
        (right_csv.resolve(), "right"),
    ]


def test_whisker_key_from_path_includes_side() -> None:
    left_root = Path("/session/whiskers/left")
    csv_path = left_root / "folderA" / "tracking.csv"
    assert whisker_key_from_path(csv_path, left_root, "left") == (
        "left__folderA__tracking"
    )


def test_resolve_whisker_csv(tmp_path: Path) -> None:
    left = tmp_path / "whiskers" / "left" / "sub"
    left.mkdir(parents=True)
    csv_path = left / "trace.csv"
    csv_path.write_text("Time,X,Y\n", encoding="utf-8")

    key = whisker_key_from_path(
        csv_path,
        tmp_path / "whiskers" / "left",
        "left",
    )
    resolved = resolve_whisker_csv(tmp_path, key)
    assert resolved == csv_path.resolve()
