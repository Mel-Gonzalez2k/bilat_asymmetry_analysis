#!/usr/bin/env python3
r"""
Generate a data.json in every session folder under
E:\bilat_asymmetry_analysis\data\Optogenetic_Activation
that contains exactly one .h5 file and exactly one *_noB.mp4 video.

Folder layout (animal / date / session):
    Optogenetic_Activation\MRN_opto15\202260527\1_whisking_first_stim_002_10Hz\

In each session folder there are several files; only two are referenced:
  - the .h5                          -> the hdf5 "filepath" entries
  - the *_noB.mp4 (the no-border video) -> the video "filepath" entry
The .avi and .csv files in the folder are ignored.

For each qualifying folder a data.json is written where:
  - entries with "format": "hdf5"     -> "filepath" = that folder's .h5 filename
  - the entry with "data_type": "video" -> "filepath" = that folder's *_noB.mp4
  - everything else in the template (channels, clocks, data_key) is unchanged

Rules (same spirit as generate_json.py):
- A folder that already has a data.json is SKIPPED (not overwritten).
- A folder with more than one .h5 or more than one *_noB.mp4 is SKIPPED and
  flagged for manual review (ambiguous which file to reference).
- A folder with only an .h5 OR only a *_noB.mp4 (not both) is SKIPPED and
  flagged for manual review.
- Folders with neither are ignored silently.

Usage:
    uv run generate_json_optoact.py            # dry run (default) - lists what would be created
    uv run generate_json_optoact.py --execute  # actually writes the data.json files
    uv run generate_json_optoact.py --root "E:\...\Optogenetic_Activation"   # override root
"""
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

import argparse
import json
import copy
import re
from pathlib import Path

DEFAULT_ROOT = Path(r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation")


def sweep_key_from_h5(h5_name: str) -> str:
    """Derive the HDF5 data_key from the .h5 filename.

    The last 4-digit group before ".h5" is the sweep number, e.g.
        ..._2026-05-27_001_0002.h5  ->  sweep_0002/digitalScans
    Falls back to sweep_0001 if no trailing 4-digit group is found.
    """
    m = re.search(r"(\d{4})\.h5$", h5_name)
    sweep = m.group(1) if m else "0001"
    return f"sweep_{sweep}/digitalScans"

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
    data_key = sweep_key_from_h5(h5_name)
    for item in doc["data"]:
        if item.get("format") == "hdf5":
            item["filepath"] = h5_name
            item["data_key"] = data_key
        elif item.get("data_type") == "video":
            item["filepath"] = mp4_name
    return doc


def scan(root: Path, overwrite: bool = False):
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    to_create = []     # (folder, h5_name, mp4_name)
    already_has = []   # folder already has data.json
    ambiguous = []     # multiple h5 or multiple _noB.mp4
    incomplete = []    # only h5 or only _noB.mp4

    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        h5_files = list(folder.glob("*.h5"))
        # ONLY the no-border videos, not every .mp4 (and never .avi)
        mp4_files = [p for p in folder.glob("*_noB.mp4")]

        if not h5_files and not mp4_files:
            continue  # not relevant, skip silently

        if (folder / "data.json").exists() and not overwrite:
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
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="Top of the Optogenetic_Activation tree to scan.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually write data.json files. Without this flag, runs as a dry run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate data.json even in folders that already have one "
                             "(use this to fix files written with a wrong data_key).")
    args = parser.parse_args()
    root = Path(args.root)

    to_create, already_has, ambiguous, incomplete = scan(root, args.overwrite)

    print(f"\n{'DRY RUN' if not args.execute else 'EXECUTING'} - "
          f"{len(to_create)} data.json file(s) to create\n")

    for folder, h5_name, mp4_name in to_create:
        print(f"{folder}")
        print(f"   h5:  {h5_name}")
        print(f"   mp4: {mp4_name}")
        print()

    if already_has:
        print(f"[info] {len(already_has)} folder(s) already have a data.json - left alone:")
        for folder in already_has:
            print(f"   {folder}")
        print()

    if ambiguous:
        print(f"[warn] {len(ambiguous)} folder(s) skipped - multiple .h5 or *_noB.mp4 files found:")
        for folder, h5_files, mp4_files in ambiguous:
            print(f"   {folder}")
            print(f"      h5:  {[f.name for f in h5_files]}")
            print(f"      mp4: {[f.name for f in mp4_files]}")
        print()

    if incomplete:
        print(f"[warn] {len(incomplete)} folder(s) skipped - only .h5 or only *_noB.mp4 present:")
        for folder, h5_files, mp4_files in incomplete:
            print(f"   {folder}  (h5: {len(h5_files)}, noB.mp4: {len(mp4_files)})")
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
