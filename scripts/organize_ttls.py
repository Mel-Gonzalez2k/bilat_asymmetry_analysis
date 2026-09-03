"""
Organize TeLC behavior files into E:\bilat_asymmetry_analysis\raw\TeLC_Silencing
structured as <Mouse>\<Date>\<Session>\

Rules:
- Mouse ID = leading WA0## token (e.g. WA024_L_R_... -> WA024)
- If a mouse has ANY folder with a _DayN suffix:
    - folders with no date-suffix at all (no _DayN) -> session "Pre_Day0"
    - folders with _DayN -> session "Post_DayN"
- If a mouse has NO _DayN suffix on any of its folders:
    - sort that mouse's dates chronologically
    - earliest date -> "Pre_Day0"
    - each later date -> "Post_Day{elapsed_days_since_Pre_Day0}"
- Only .h5 and .csv files are copied (not moved) from each source folder.
- Folders that don't parse cleanly (no mouse ID or no date) are skipped and reported.

Usage:
    python organize_telc_files.py                 # dry run (default) - prints mapping only
    python organize_telc_files.py --execute        # actually copies files
"""

import argparse
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SOURCE_ROOT = Path(r"D:\Whisker_Asymmetry\3_MRN_TelC_cohort\2026_TelC_behavior\TTLs_202604")
DEST_ROOT = Path(r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing")

MOUSE_RE = re.compile(r"^(WA\d+)")
DAY_SUFFIX_RE = re.compile(r"_Day(\d+)$")
DATE_RE = re.compile(r"(20\d{6})")  # e.g. 20260322


def parse_folder(name: str):
    """Return (mouse_id, date_str, day_suffix or None) or None if unparseable."""
    mouse_match = MOUSE_RE.match(name)
    if not mouse_match:
        return None
    mouse_id = mouse_match.group(1)

    day_match = DAY_SUFFIX_RE.search(name)
    day_suffix = int(day_match.group(1)) if day_match else None

    date_match = DATE_RE.search(name)
    if not date_match:
        return (mouse_id, None, day_suffix)
    date_str = date_match.group(1)

    return (mouse_id, date_str, day_suffix)


def build_mapping():
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Source root not found: {SOURCE_ROOT}")

    folders = [p for p in SOURCE_ROOT.iterdir() if p.is_dir()]

    parsed = []   # list of (path, mouse_id, date_str, day_suffix)
    skipped = []

    for folder in folders:
        result = parse_folder(folder.name)
        if result is None or result[1] is None:
            skipped.append(folder)
            continue
        mouse_id, date_str, day_suffix = result
        parsed.append((folder, mouse_id, date_str, day_suffix))

    # group by mouse
    by_mouse = defaultdict(list)
    for item in parsed:
        by_mouse[item[1]].append(item)

    mapping = []  # (source_folder, dest_path)

    for mouse_id, items in by_mouse.items():
        has_day_suffix = any(day_suffix is not None for _, _, _, day_suffix in items)

        if has_day_suffix:
            for folder, _, date_str, day_suffix in items:
                session = "Pre_Day0" if day_suffix is None else f"Post_Day{day_suffix}"
                dest = DEST_ROOT / mouse_id / date_str / session
                mapping.append((folder, dest))
        else:
            # no DayN suffix anywhere for this mouse -> sort dates chronologically
            dated_items = sorted(items, key=lambda x: x[2])  # sort by date_str
            baseline_date_str = dated_items[0][2]
            baseline_date = datetime.strptime(baseline_date_str, "%Y%m%d")

            for folder, _, date_str, _ in dated_items:
                this_date = datetime.strptime(date_str, "%Y%m%d")
                elapsed = (this_date - baseline_date).days
                session = "Pre_Day0" if elapsed == 0 else f"Post_Day{elapsed}"
                dest = DEST_ROOT / mouse_id / date_str / session
                mapping.append((folder, dest))

    return mapping, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually copy files. Without this flag, runs as a dry run.")
    args = parser.parse_args()

    mapping, skipped = build_mapping()

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} - {len(mapping)} folders mapped\n")
    print(f"{'SOURCE':70s} -> DEST")
    print("-" * 140)
    for src, dest in sorted(mapping, key=lambda x: str(x[1])):
        print(f"{src.name:70s} -> {dest}")

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} folder(s) - could not parse mouse ID or date:")
        for s in skipped:
            print(f"   {s.name}")

    if not args.execute:
        print("\nThis was a DRY RUN. No files were copied.")
        print("Review the mapping above, then re-run with --execute to actually copy files.")
        return

    print("\nCopying files...\n")
    copied_count = 0
    for src, dest in mapping:
        dest.mkdir(parents=True, exist_ok=True)
        for ext in ("*.h5", "*.csv"):
            for f in src.glob(ext):
                dest_file = dest / f.name
                if dest_file.exists():
                    print(f"   SKIP (already exists): {dest_file}")
                    continue
                shutil.copy2(f, dest_file)
                print(f"   copied: {f.name} -> {dest}")
                copied_count += 1

    print(f"\nDone. Copied {copied_count} file(s).")


if __name__ == "__main__":
    main()