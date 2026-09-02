"""Discover left- and right-whisker tracking CSV files for TeLC sessions."""

from __future__ import annotations

from pathlib import Path

_WHISKER_KEY_SEP = "__"
_WHISKER_SIDES = ("left", "right")


def infer_whisker_side(whisker_csv: str | Path) -> str:
    """Infer whisker side from a path under ``whiskers/left`` or ``whiskers/right``.

    Parameters
    ----------
    whisker_csv
        Path to a whisker tracking CSV.

    Returns
    -------
    str
        ``left`` or ``right``.

    Raises
    ------
    ValueError
        If the path is not under a recognized whisker-side folder.
    """
    path = Path(whisker_csv).resolve().as_posix().lower()
    if "/whiskers/left/" in path:
        return "left"
    if "/whiskers/right/" in path:
        return "right"
    msg = f"Cannot infer whisker side from path: {whisker_csv}"
    raise ValueError(msg)


def whiskers_side_dir(dynamic_session_dir: str | Path, side: str) -> Path:
    """Return ``whiskers/{side}`` for one dynamic session."""
    return Path(dynamic_session_dir) / "whiskers" / side


def left_whiskers_dir(dynamic_session_dir: str | Path) -> Path:
    """Return the ``whiskers/left`` directory for one dynamic session."""
    return whiskers_side_dir(dynamic_session_dir, "left")


def whisker_key_from_path(
    csv_path: Path,
    side_root: Path,
    side: str,
) -> str:
    """Encode a whisker CSV path as a stable Snakemake wildcard key.

    Parameters
    ----------
    csv_path
        Absolute or relative path to a whisker tracking CSV.
    side_root
        Root directory ``whiskers/{side}`` for the session.
    side
        Whisker side: ``left`` or ``right``.

    Returns
    -------
    str
        ``{side}__`` plus the relative path under ``side_root`` with path
        separators replaced by ``__`` and the ``.csv`` suffix removed.
    """
    relative = csv_path.resolve().relative_to(side_root.resolve())
    parts = list(relative.parts)
    if parts[-1].lower().endswith(".csv"):
        parts[-1] = parts[-1][:-4]
    relative_key = _WHISKER_KEY_SEP.join(parts)
    return f"{side}{_WHISKER_KEY_SEP}{relative_key}"


def discover_side_whisker_csvs(
    dynamic_session_dir: str | Path,
    side: str,
) -> list[Path]:
    """Return every ``.csv`` under ``whiskers/{side}`` for one session."""
    side_dir = whiskers_side_dir(dynamic_session_dir, side)
    if not side_dir.is_dir():
        return []
    return sorted(path.resolve() for path in side_dir.rglob("*.csv") if path.is_file())


def discover_left_whisker_csvs(dynamic_session_dir: str | Path) -> list[Path]:
    """Return every ``.csv`` under ``whiskers/left`` for one session."""
    return discover_side_whisker_csvs(dynamic_session_dir, "left")


def discover_whisker_csvs(dynamic_session_dir: str | Path) -> list[tuple[Path, str]]:
    """Return every whisker CSV under ``whiskers/left`` and ``whiskers/right``.

    Returns
    -------
    list[tuple[Path, str]]
        Sorted ``(csv_path, side)`` pairs.
    """
    discovered: list[tuple[Path, str]] = []
    for side in _WHISKER_SIDES:
        for csv_path in discover_side_whisker_csvs(dynamic_session_dir, side):
            discovered.append((csv_path, side))
    return discovered


def resolve_whisker_csv(
    dynamic_session_dir: str | Path,
    whisker_key: str,
) -> Path:
    """Resolve a whisker key back to its source CSV path.

    Parameters
    ----------
    dynamic_session_dir
        Session directory under ``data/external/dynamic``.
    whisker_key
        Key produced by :func:`whisker_key_from_path`.

    Returns
    -------
    Path
        Matching whisker CSV path.

    Raises
    ------
    FileNotFoundError
        If no CSV matches ``whisker_key``.
    """
    for csv_path, side in discover_whisker_csvs(dynamic_session_dir):
        side_root = whiskers_side_dir(dynamic_session_dir, side)
        if whisker_key_from_path(csv_path, side_root, side) == whisker_key:
            return csv_path
    msg = (
        f"No whisker CSV found for key {whisker_key!r} under "
        f"{Path(dynamic_session_dir) / 'whiskers'}"
    )
    raise FileNotFoundError(msg)
