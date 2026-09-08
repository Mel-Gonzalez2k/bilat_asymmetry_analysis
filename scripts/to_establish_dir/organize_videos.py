"""
Organize TeLC behavior videos into the existing
E:\bilat_asymmetry_analysis\raw\TeLC_Silencing\<Mouse>\<Date>\<Session>\ structure.

Unlike the h5/csv folders, video filenames do NOT contain a date - only a mouse ID,
an index number, and either "pre" or "DayN". Because of that, this script does NOT
invent a date. Instead, it looks up the date folder that ALREADY EXISTS on disk for
that mouse + session (created earlier by organize_telc_files.py / move_files.py) and
copies the video there.

Rules:
- Mouse ID = leading WA0## token (e.g. WA024_L_R_... -> WA024)
- If the filename contains "_DayN" -> session = "Post_DayN"
- If it doesn't (e.g. "..._pre_TelC0_noB_rot180.mp4") -> session = "Pre_Day0"
- KNOWN SAVE TYPO: a take/version number sometimes gets fused directly onto the day
  digits with no separator (e.g. "Day70", "Day101", "Day110"). This script strips
  that fused digit by checking against the known real day values (KNOWN_DAYS below):
  if the captured digit run isn't itself a known day, but dropping its last digit
  IS a known day, the last digit is treated as a fused take number and dropped.
  e.g. "Day70" -> 7, "Day101" -> 10, "Day110" -> 11, "Day130" -> 13.
  Filenames where the day digits are already a known day and are followed by an
  underscore-separated tag (e.g. "Day10_notail_2", "Day7_2") are left alone.
- Destination = E:\...\TeLC_Silencing\<Mouse>\<ExistingDateFolder>\<Session>\
  found by searching for a folder already named <Session> under that mouse's directory.
- If no matching session folder exists yet for that mouse (e.g. WA024 Day11/Day13),
  the video is SKIPPED and listed for manual handling - no folder is guessed/created.
- Multiple videos for the same session (different takes/angles) are ALL copied,
  keeping their original filenames.
- Only video files matching VIDEO_EXT are processed (subfolders, like a masks/frames
  folder sitting alongside the videos, are ignored since this only globs top-level files).

Usage:
    python organize_videos.py                 # dry run (default) - prints mapping only
    python organize_videos.py --execute        # actually copies files
"""

import argparse
import re
import shutil
from pathlib import Path
from collections import defaultdict

SOURCE_ROOT = Path(r"D:\Whisker_Asymmetry\3_MRN_TelC_cohort\2026_TelC_behavior\3_202604_rotated_nob_frames_videos")
DEST_ROOT = Path(r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing")
VIDEO_EXT = "*.mp4"

MOUSE_RE = re.compile(r"^(WA\d+)")
DAY_RE = re.compile(r"_Day(\d+)")
KNOWN_DAYS = {"0", "7", "8", "10", "11", "13"}


def parse_video(name: str):
    """Return (mouse_id, session) or None if mouse ID can't be parsed."""
    mouse_match = MOUSE_RE.match(name)
    if not mouse_match:
        return None
    mouse_id = mouse_match.group(1)

    day_match = DAY_RE.search(name)
    if day_match is None:
        return mouse_id, "Pre_Day0"

    day_str = day_match.group(1)
    if day_str not in KNOWN_DAYS and day_str[:-1] in KNOWN_DAYS:
        # trailing digit is a fused take-number typo - drop it
        day_str = day_str[:-1]

    return mouse_id, f"Post_Day{day_str}"


def find_session_folder(mouse_id: str, session: str):
    """Look for an already-existing <Mouse>/<Date>/<Session> folder. Returns Path or None."""
    mouse_dir = DEST_ROOT / mouse_id
    if not mouse_dir.exists():
        return None
    matches = list(mouse_dir.glob(f"*/{session}"))
    if len(matches) == 1:
        return matches[0]
    return None  # none found, or ambiguous (>1) - treat both as "not found" for safety


def build_mapping():
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Source root not found: {SOURCE_ROOT}")

    videos = sorted(SOURCE_ROOT.glob(VIDEO_EXT))

    mapping = []     # (video_path, dest_folder)
    unparsed = []    # couldn't extract mouse ID
    no_folder = []   # parsed fine, but no matching destination folder exists yet

    for video in videos:
        result = parse_video(video.name)
        if result is None:
            unparsed.append(video)
            continue
        mouse_id, session = result
        dest_folder = find_session_folder(mouse_id, session)
        if dest_folder is None:
            no_folder.append((video, mouse_id, session))
            continue
        mapping.append((video, dest_folder))

    return mapping, unparsed, no_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually copy files. Without this flag, runs as a dry run.")
    args = parser.parse_args()

    mapping, unparsed, no_folder = build_mapping()

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} - {len(mapping)} video(s) mapped\n")
    print(f"{'VIDEO':55s} -> DEST")
    print("-" * 140)
    for src, dest in sorted(mapping, key=lambda x: str(x[1])):
        print(f"{src.name:55s} -> {dest}")

    if no_folder:
        print(f"\n⚠️  {len(no_folder)} video(s) skipped - no matching Mouse/Date/Session folder exists yet:")
        for video, mouse_id, session in no_folder:
            print(f"   {video.name:55s} (mouse={mouse_id}, session={session})")

    if unparsed:
        print(f"\n⚠️  {len(unparsed)} video(s) skipped - could not parse a mouse ID:")
        for v in unparsed:
            print(f"   {v.name}")

    if not args.execute:
        print("\nThis was a DRY RUN. No files were copied.")
        print("Review the mapping above, then re-run with --execute to actually copy files.")
        return

    print("\nCopying files...\n")
    copied_count = 0
    for src, dest in mapping:
        dest_file = dest / src.name
        if dest_file.exists():
            print(f"   SKIP (already exists): {dest_file}")
            continue
        shutil.copy2(src, dest_file)
        print(f"   copied: {src.name} -> {dest}")
        copied_count += 1

    print(f"\nDone. Copied {copied_count} file(s).")


if __name__ == "__main__":
    main()
