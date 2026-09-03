"""
Organize files from E:\bilat_asymmetry_analysis\data_to_organize into
E:\bilat_asymmetry_analysis\scratch_space\TeLC_Silencing\<Mouse>\<Date>\<Session>\

Source folders look like "WA020_R_MRN_TelC_03_Day8" and contain a mix of files:
  - left_*.csv   -> copied into <Session>\left\
  - right_*.csv  -> copied into <Session>\right\
  - *.json       -> copied directly into <Session>\
  - *.h5         -> copied directly into <Session>\
  - *.mp4        -> left alone, not copied
  - anything else (e.g. .log files) -> left alone, just reported

The date for each session isn't in the source folder name, so it's looked up from
the already-organized E:\bilat_asymmetry_analysis\raw\TeLC_Silencing\<Mouse>\<Date>\<Session>\
folder (same lookup approach as organize_videos.py). If no matching date folder is
found there either, the source folder is skipped and flagged - no date is guessed.

Rules:
- Mouse ID = leading WA0## token (e.g. WA024_L_R_M_... -> WA024)
- "_Day0" -> session = "Pre_Day0"
- "_DayN" (N>0) -> session = "Post_DayN"
- data_to_organize is READ ONLY - nothing is ever written there.
- Destination is scratch_space\TeLC_Silencing - created fresh as needed
  (left/right subfolders created automatically).

Usage:
    python organize_left_right.py             # dry run (default) - prints mapping only
    python organize_left_right.py --execute   # actually copies files
"""

import argparse
import re
import shutil
from pathlib import Path

SOURCE_ROOT = Path(r"E:\bilat_asymmetry_analysis\data_to_organize")
RAW_ROOT = Path(r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing")
SCRATCH_ROOT = Path(r"E:\bilat_asymmetry_analysis\scratch_space\TeLC_Silencing")

MOUSE_RE = re.compile(r"^(WA\d+)")
DAY_RE = re.compile(r"_Day(\d+)")


def parse_source_folder(name: str):
    """Return (mouse_id, session) or None if mouse ID can't be parsed."""
    mouse_match = MOUSE_RE.match(name)
    if not mouse_match:
        return None
    mouse_id = mouse_match.group(1)

    day_match = DAY_RE.search(name)
    if day_match is None:
        return None  # no day info at all - can't determine session
    day_str = day_match.group(1)
    session = "Pre_Day0" if day_str == "0" else f"Post_Day{day_str}"

    return mouse_id, session


def find_date_for_session(mouse_id: str, session: str):
    """Look up the real date folder from the already-organized raw/ tree. Returns str or None."""
    mouse_dir = RAW_ROOT / mouse_id
    if not mouse_dir.exists():
        return None
    matches = list(mouse_dir.glob(f"*/{session}"))
    if len(matches) == 1:
        return matches[0].parent.name  # the date folder name
    return None  # none found, or ambiguous


def classify_files(folder: Path):
    """Return dict: {'left': [...], 'right': [...], 'copy_direct': [...], 'ignored': [...]}"""
    result = {"left": [], "right": [], "copy_direct": [], "ignored": []}
    for f in folder.iterdir():
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        if name_lower.startswith("left_") and f.suffix.lower() == ".csv":
            result["left"].append(f)
        elif name_lower.startswith("right_") and f.suffix.lower() == ".csv":
            result["right"].append(f)
        elif f.suffix.lower() in (".json", ".h5"):
            result["copy_direct"].append(f)
        else:
            result["ignored"].append(f)  # includes .mp4 and .log - left alone
    return result


def build_mapping():
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Source root not found: {SOURCE_ROOT}")

    source_folders = sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir())

    mapping = []      # (session_dir, {left:[...], right:[...], copy_direct:[...]})
    unparsed = []      # couldn't extract mouse ID / day
    no_date = []       # parsed fine, but no matching date found in raw/

    for folder in source_folders:
        result = parse_source_folder(folder.name)
        if result is None:
            unparsed.append(folder)
            continue
        mouse_id, session = result

        date_str = find_date_for_session(mouse_id, session)
        if date_str is None:
            no_date.append((folder, mouse_id, session))
            continue

        session_dir = SCRATCH_ROOT / mouse_id / date_str / session
        files = classify_files(folder)
        mapping.append((folder, session_dir, files))

    return mapping, unparsed, no_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually copy files. Without this flag, runs as a dry run.")
    args = parser.parse_args()

    mapping, unparsed, no_date = build_mapping()

    total_left = sum(len(f["left"]) for _, _, f in mapping)
    total_right = sum(len(f["right"]) for _, _, f in mapping)
    total_direct = sum(len(f["copy_direct"]) for _, _, f in mapping)

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} - "
          f"{len(mapping)} source folder(s) mapped "
          f"({total_left} left csv, {total_right} right csv, {total_direct} json/h5)\n")

    for folder, session_dir, files in mapping:
        print(f"{folder.name}")
        print(f"   -> {session_dir}")
        for f in files["left"]:
            print(f"      left\\{f.name}")
        for f in files["right"]:
            print(f"      right\\{f.name}")
        for f in files["copy_direct"]:
            print(f"      {f.name}")
        if files["ignored"]:
            ignored_names = ", ".join(f.name for f in files["ignored"])
            print(f"      (ignored: {ignored_names})")
        print()

    if no_date:
        print(f"⚠️  {len(no_date)} source folder(s) skipped - no matching date found in {RAW_ROOT}:")
        for folder, mouse_id, session in no_date:
            print(f"   {folder.name:45s} (mouse={mouse_id}, session={session})")
        print()

    if unparsed:
        print(f"⚠️  {len(unparsed)} source folder(s) skipped - could not parse mouse ID / day:")
        for f in unparsed:
            print(f"   {f.name}")
        print()

    if not args.execute:
        print("This was a DRY RUN. No files were copied.")
        print("Review the mapping above, then re-run with --execute to actually copy files.")
        return

    print("Copying files...\n")
    copied_count = 0
    for folder, session_dir, files in mapping:
        left_dir = session_dir / "left"
        right_dir = session_dir / "right"

        if files["left"]:
            left_dir.mkdir(parents=True, exist_ok=True)
        if files["right"]:
            right_dir.mkdir(parents=True, exist_ok=True)
        if files["copy_direct"]:
            session_dir.mkdir(parents=True, exist_ok=True)

        for f in files["left"]:
            dest_file = left_dir / f.name
            if dest_file.exists():
                print(f"   SKIP (already exists): {dest_file}")
                continue
            shutil.copy2(f, dest_file)
            print(f"   copied: {f.name} -> {left_dir}")
            copied_count += 1

        for f in files["right"]:
            dest_file = right_dir / f.name
            if dest_file.exists():
                print(f"   SKIP (already exists): {dest_file}")
                continue
            shutil.copy2(f, dest_file)
            print(f"   copied: {f.name} -> {right_dir}")
            copied_count += 1

        for f in files["copy_direct"]:
            dest_file = session_dir / f.name
            if dest_file.exists():
                print(f"   SKIP (already exists): {dest_file}")
                continue
            shutil.copy2(f, dest_file)
            print(f"   copied: {f.name} -> {session_dir}")
            copied_count += 1

    print(f"\nDone. Copied {copied_count} file(s).")


if __name__ == "__main__":
    main()