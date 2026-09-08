#!/usr/bin/env python3
r"""
Generate one alignment JSON per (.h5, .mp4) recording pair, following the
structure of data_MRN12.json.

Pairing logic (from the example pair):
    h5  : <base>_2026-05-15_001_0001.h5
    mp4 : <base>0_noB.mp4
where <base> is e.g. "MRN_opto12_M_L_20260517_100ms_1s_int_5x"
(or "..._5x_clean_wheel").

The JSON is named data_<mouse>.json, where <mouse> is the MRN_optoNN token
(e.g. data_MRN_opto12.json). If a mouse has more than one recording, the
condition is appended so files don't collide (e.g. data_MRN_opto12_100ms.json).

Usage:
    python make_jsons.py "E:\bilat_asymmetry_analysis\raw\Sept_test_ttl\opto_testing_ttl_alignment"

If no directory is given, the current working directory is used.
Nothing is overwritten unless you pass --overwrite.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

H5_SUFFIX = "_2026-05-15_001_0001.h5"
MP4_SUFFIX = "0_noB.mp4"
DATA_KEY = "sweep_0001/digitalScans"


def h5_base(name: str):
    """Return the shared base for an .h5 file, or None if it doesn't match."""
    if name.endswith(H5_SUFFIX):
        return name[: -len(H5_SUFFIX)]
    return None


def mp4_base(name: str):
    """Return the shared base for an .mp4 file, or None if it doesn't match."""
    if name.endswith(MP4_SUFFIX):
        return name[: -len(MP4_SUFFIX)]
    return None


def mouse_token(base: str) -> str:
    """Pull the MRN_optoNN token out of a base name."""
    m = re.match(r"(MRN_opto\d+)", base)
    return m.group(1) if m else base


def condition_token(base: str, mouse: str) -> str:
    """Everything after the mouse+sex+side+date, used to disambiguate names."""
    # base like MRN_opto12_M_L_20260517_100ms_1s_int_5x[_clean_wheel]
    # strip the mouse token and the following _<sex>_<side>_<date>_
    m = re.match(rf"{re.escape(mouse)}_[MF]_[LR]_\d+_(.+)", base)
    return m.group(1) if m else base


def build_json(h5_name: str, mp4_name: str) -> dict:
    """Assemble the JSON dict for one pair, matching data_MRN12.json exactly."""
    return {
        "clocks": ["master", "time"],
        "data": [
            {
                "name": "master",
                "format": "hdf5",
                "data_type": "time",
                "filepath": h5_name,
                "data_key": DATA_KEY,
                "time_layout": "identity",
            },
            {
                "name": "cam_ttl",
                "format": "hdf5",
                "data_type": "digital_event",
                "filepath": h5_name,
                "data_key": DATA_KEY,
                "channel": 0,
                "transition": "rising",
                "clock": "master",
            },
            {
                "format": "derived",
                "data_type": "time",
                "name": "time",
                "source_timeframe": "master",
                "source_series": "cam_ttl",
                "source_type": "event",
            },
            {
                "name": "pulse_pal_intervals",
                "format": "hdf5",
                "data_type": "digital_interval",
                "filepath": h5_name,
                "data_key": DATA_KEY,
                "channel": 1,
                "transition": "rising",
                "clock": "master",
            },
            {
                "filepath": mp4_name,
                "data_type": "video",
                "name": "media",
            },
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default=".",
                    help="Folder containing the .h5 and .mp4 files.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing JSON files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written without writing.")
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    names = [p.name for p in root.iterdir() if p.is_file()]

    h5_by_base = {b: n for n in names if (b := h5_base(n)) is not None}
    mp4_by_base = {b: n for n in names if (b := mp4_base(n)) is not None}

    bases = sorted(set(h5_by_base) & set(mp4_by_base))

    # Report anything that couldn't be paired, so silent drops don't happen.
    lonely_h5 = sorted(set(h5_by_base) - set(mp4_by_base))
    lonely_mp4 = sorted(set(mp4_by_base) - set(h5_by_base))
    for b in lonely_h5:
        print(f"  [warn] h5 with no matching mp4: {h5_by_base[b]}")
    for b in lonely_mp4:
        print(f"  [warn] mp4 with no matching h5: {mp4_by_base[b]}")

    # Decide output names; disambiguate mice with multiple recordings.
    by_mouse = defaultdict(list)
    for b in bases:
        by_mouse[mouse_token(b)].append(b)

    written = 0
    for b in bases:
        mouse = mouse_token(b)
        if len(by_mouse[mouse]) == 1:
            out_name = f"data_{mouse}.json"
        else:
            out_name = f"data_{mouse}_{condition_token(b, mouse)}.json"

        out_path = root / out_name
        payload = build_json(h5_by_base[b], mp4_by_base[b])

        print(f"  {out_name}")
        print(f"      h5 : {h5_by_base[b]}")
        print(f"      mp4: {mp4_by_base[b]}")

        if args.dry_run:
            continue
        if out_path.exists() and not args.overwrite:
            print(f"      [skip] exists (use --overwrite to replace)")
            continue

        with open(out_path, "w", newline="\r\n") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        written += 1

    print(f"\nPairs found: {len(bases)}   JSON written: {written}"
          + ("  (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
