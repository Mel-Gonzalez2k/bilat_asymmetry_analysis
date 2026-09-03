"""
Generate a data.json in every folder under
E:\bilat_asymmetry_analysis\raw\TeLC_Silencing
that contains exactly one .h5 file and exactly one .mp4 file.

The template below is copied from the existing WA020\20260322\Pre_Day0\data.json.
For each qualifying folder, a new data.json is written where:
  - any entry with "format": "hdf5"   -> "filepath" is set to that folder's .h5 filename
  - any entry with "data_type": "video" -> "filepath" is set to that folder's .mp4 filename
  - everything else in the template (channels, clocks, data_key, etc.) is left unchanged

Rules:
- A folder with a data.json already present is SKIPPED (not overwritten).
- A folder with more than one .h5 or more than one .mp4 is SKIPPED and flagged
  for manual review (ambiguous which file to reference).
- A folder with only an .h5 OR only an .mp4 (not both) is SKIPPED and flagged
  for manual review (can't build a complete data.json).
- Folders with neither an .h5 nor an .mp4 are ignored silently (not relevant).

Usage:
    python generate_data_json.py             # dry run (default) - lists what would be created
    python generate_data_json.py --execute   # actually writes the data.json files
"""

import argparse
import json
import copy
from pathlib import Path

ROOT = Path(r"E:\bilat_asymmetry_analysis\raw\TeLC_Silencing")

TEMPLATE = {
    "clocks": ["master", "time"],
    "data": [
        {
            "name": "master",
            "format": "hdf5",
            "data_type": "time",
            "filepath": "PLACEHOLDER.h5",
            "data_key": "sweep_0001/digitalScans",
            "time_layout": "identity"
        },
        {
            "name": "cam_ttl",
            "format": "hdf5",
            "data_type": "digital_event",
            "filepath": "PLACEHOLDER.h5",
            "data_key": "sweep_0001/digitalScans",
            "channel": 0,
            "transition": "rising",
            "clock": "master"
        },
        {
            "format": "derived",
            "data_type": "time",
            "name": "time",
            "source_timeframe": "master",
            "source_series": "cam_ttl",
            "source_type": "event"
        },
        {
            "name": "pulse_pal_intervals",
            "format": "hdf5",
            "data_type": "digital_interval",
            "filepath": "PLACEHOLDER.h5",
            "data_key": "sweep_0001/digitalScans",
            "channel": 1,
            "transition": "rising",
            "clock": "master"
        },
        {
            "filepath": "PLACEHOLDER.mp4",
            "data_type": "video",
            "name": "media"
        }
    ]
}


def build_data_json(h5_name: str, mp4_name: str) -> dict:
    doc = copy.deepcopy(TEMPLATE)
    for item in doc["data"]:
        if item.get("format") == "hdf5":
            item["filepath"] = h5_name
        elif item.get("data_type") == "video":
            item["filepath"] = mp4_name
    return doc


def scan():
    if not ROOT.exists():
        raise SystemExit(f"Root not found: {ROOT}")

    to_create = []     # (folder, h5_name, mp4_name)
    already_has = []   # folder already has data.json
    ambiguous = []     # multiple h5 or mp4
    incomplete = []    # only h5 or only mp4

    for folder in sorted(p for p in ROOT.rglob("*") if p.is_dir()):
        h5_files = list(folder.glob("*.h5"))
        mp4_files = list(folder.glob("*.mp4"))

        if not h5_files and not mp4_files:
            continue  # not relevant, skip silently

        if (folder / "data.json").exists():
            already_has.append(folder)
            continue

        if len(h5_files) > 1 or len(mp4_files) > 1:
            ambiguous.append((folder, h5_files, mp4_files))
            continue

        if len(h5_files) == 1 and len(mp4_files) == 1:
            to_create.append((folder, h5_files[0].name, mp4_files[0].name))
        else:
            incomplete.append((folder, h5_files, mp4_files))

    return to_create, already_has, ambiguous, incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                         help="Actually write data.json files. Without this flag, runs as a dry run.")
    args = parser.parse_args()

    to_create, already_has, ambiguous, incomplete = scan()

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} - "
          f"{len(to_create)} data.json file(s) to create\n")

    for folder, h5_name, mp4_name in to_create:
        print(f"{folder}")
        print(f"   h5:  {h5_name}")
        print(f"   mp4: {mp4_name}")
        print()

    if already_has:
        print(f"ℹ️  {len(already_has)} folder(s) already have a data.json - left alone:")
        for folder in already_has:
            print(f"   {folder}")
        print()

    if ambiguous:
        print(f"⚠️  {len(ambiguous)} folder(s) skipped - multiple .h5 or .mp4 files found:")
        for folder, h5_files, mp4_files in ambiguous:
            print(f"   {folder}")
            print(f"      h5:  {[f.name for f in h5_files]}")
            print(f"      mp4: {[f.name for f in mp4_files]}")
        print()

    if incomplete:
        print(f"⚠️  {len(incomplete)} folder(s) skipped - only .h5 or only .mp4 present:")
        for folder, h5_files, mp4_files in incomplete:
            print(f"   {folder}  (h5: {len(h5_files)}, mp4: {len(mp4_files)})")
        print()

    if not args.execute:
        print("This was a DRY RUN. No files were written.")
        print("Review the list above, then re-run with --execute to actually write data.json files.")
        return

    print("Writing data.json files...\n")
    written_count = 0
    for folder, h5_name, mp4_name in to_create:
        doc = build_data_json(h5_name, mp4_name)
        dest_file = folder / "data.json"
        with open(dest_file, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"   wrote: {dest_file}")
        written_count += 1

    print(f"\nDone. Wrote {written_count} file(s).")


if __name__ == "__main__":
    main()