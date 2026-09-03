"""
Delete the .avi video files that were previously copied into
E:\bilat_asymmetry_analysis\raw\TeLC_Silencing\<Mouse>\<Date>\<Session>\
by an earlier (incorrect-source) run of organize_videos.py.

Only *.avi files are touched - .h5 and .csv files are never removed.

Usage:
    python clean_old_videos.py             # dry run - lists what would be deleted
    python clean_old_videos.py --execute   # actually deletes them
"""

import argparse
from pathlib import Path

DEST_ROOT = Path(r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete files. Without this flag, runs as a dry run.")
    args = parser.parse_args()

    if not DEST_ROOT.exists():
        raise SystemExit(f"Destination root not found: {DEST_ROOT}")

    avi_files = sorted(DEST_ROOT.rglob("*.avi"))

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} - {len(avi_files)} .avi file(s) found\n")
    for f in avi_files:
        print(f"   {f}")

    if not args.execute:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("Review the list above, then re-run with --execute to actually delete them.")
        return

    print("\nDeleting...\n")
    for f in avi_files:
        f.unlink()
        print(f"   deleted: {f}")

    print(f"\nDone. Deleted {len(avi_files)} file(s).")


if __name__ == "__main__":
    main()
