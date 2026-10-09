#!/usr/bin/env python3
r"""
Build a WhiskerToolbox data JSON for each session folder.

For every folder it:
  * finds the video (*.mp4), the acquisition .h5 (the .h5 that is NOT
    *_line_masks.h5), and the two raw line CSVs (left_whisker*.csv /
    right_whisker*.csv, excluding the *_angle* and *_filt* files),
  * writes a {"clocks":[...], "data":[...]} JSON that:
      - uses the acquisition h5's sweep_XXXX/digitalScans as the master clock,
      - derives the camera-frame "time" clock from the cam TTL (channel 0),
      - points pulse_pal_intervals at channel 1 (WhiskerToolbox extracts the
        on/off intervals itself, same as your example data.json),
      - loads the RIGHT whisker line from CSV  -> green   #00FF00,
      - loads the LEFT  whisker line from CSV  -> magenta #FF00FF,
      - loads the video as "media".
  * (optional) opens the acquisition h5 and prints how many cam frames and
    pulse-pal pulses it sees, as a sanity check that channel 0/1 are right.

The JSON is written next to the video as  <video_stem>_data1.json
(any existing file of that name is backed up to  *.bak  first).

Usage
-----
    # all three default folders:
    uv run make_wt_json.py

    # one folder:
    uv run make_wt_json.py --dir "E:\...\MRN8_output\100ms"

    # just show what it found + the detected pulses, write nothing:
    uv run make_wt_json.py --dry-run

Dependencies (only needed for the --validate sanity read): h5py, numpy.
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy"]
# ///

import argparse
import glob
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
DEFAULT_DIRS = [
    r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\MRN8_output\100ms",
    r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\Brooke_20260216_MRN_opto5_F_8_2_25_L_100ms",
    r"E:\bilat_asymmetry_analysis\data\Optogenetic_Activation\Brooke_20260219_MRN_opto6_F_8_9_25_R_100ms",
]

RIGHT_COLOR = "#00FF00"   # green   (Green=255)
LEFT_COLOR  = "#FF00FF"   # magenta (Red=255, Blue=255)
CAM_CHANNEL = 0           # camera TTL   (bit 0 of digitalScans)
PULSE_CHANNEL = 1         # pulse-pal TTL (bit 1 of digitalScans)
# ----------------------------------------------------------------------------


def pick_one(cands, what, folder):
    if not cands:
        raise FileNotFoundError(f"  [!] no {what} found in {folder}")
    if len(cands) > 1:
        print(f"  [!] {len(cands)} candidates for {what}; using {cands[0].name}")
        for c in cands[1:]:
            print(f"        (ignored: {c.name})")
    return cands[0]


def find_video(folder: Path) -> Path:
    cands = [Path(p) for p in glob.glob(str(folder / "*.mp4"))]
    return pick_one(sorted(cands), "video (*.mp4)", folder)


def find_acq_h5(folder: Path) -> Path:
    h5s = [Path(p) for p in glob.glob(str(folder / "*.h5"))]
    acq = [p for p in h5s if "line_masks" not in p.name.lower()]
    # prefer a file ending in a 4-digit run number (the WaveSurfer acquisition)
    runlike = [p for p in acq if re.search(r"_\d{4}\.h5$", p.name)]
    chosen = sorted(runlike) or sorted(acq)
    return pick_one(chosen, "acquisition .h5 (non line_masks)", folder)


def find_line_csv(folder: Path, side: str) -> Path:
    cands = []
    for p in glob.glob(str(folder / f"{side}_whisker*.csv")):
        n = Path(p).name.lower()
        if "angle" in n or "filt" in n or "trace" in n:
            continue
        cands.append(Path(p))
    return pick_one(sorted(cands), f"{side}_whisker line CSV", folder)


def sweep_key(acq_name: str) -> str:
    """digitalScans group: trailing _NNNN.h5 run number, else last 4-digit run."""
    m = re.search(r"_(\d{4})\.h5$", acq_name)
    if m:
        run = m.group(1)
    else:
        g = re.findall(r"\d{4}", acq_name)
        run = g[-1] if g else "0001"
    return f"sweep_{run}/digitalScans"


def build_json(video, acq_h5, right_csv, left_csv, dkey):
    """Return the WhiskerToolbox config dict (filepaths are basenames)."""
    acq = acq_h5.name
    return {
        "clocks": ["master", "time"],
        "data": [
            {   # master acquisition clock (250 kHz samples of digitalScans)
                "name": "master",
                "format": "hdf5",
                "data_type": "time",
                "filepath": acq,
                "data_key": dkey,
                "time_layout": "identity",
            },
            {   # camera TTL -> one rising edge per video frame
                "name": "cam_ttl",
                "format": "hdf5",
                "data_type": "digital_event",
                "filepath": acq,
                "data_key": dkey,
                "channel": CAM_CHANNEL,
                "transition": "rising",
                "clock": "master",
            },
            {   # camera-frame timebase that the whisker lines live on
                "format": "derived",
                "data_type": "time",
                "name": "time",
                "source_timeframe": "master",
                "source_series": "cam_ttl",
                "source_type": "event",
            },
            {   # pulse-pal stim intervals (WhiskerToolbox extracts on/off itself)
                "name": "pulse_pal_intervals",
                "format": "hdf5",
                "data_type": "digital_interval",
                "filepath": acq,
                "data_key": dkey,
                "channel": PULSE_CHANNEL,
                "transition": "rising",
                "clock": "master",
            },
            {   # RIGHT whisker line from CSV -> green
                "filepath": right_csv.name,
                "data_type": "line",
                "name": "right_whisker",
                "format": "csv",
                "color": RIGHT_COLOR,
                "clock": "time",
            },
            {   # LEFT whisker line from CSV -> magenta
                "filepath": left_csv.name,
                "data_type": "line",
                "name": "left_whisker",
                "format": "csv",
                "color": LEFT_COLOR,
                "clock": "time",
            },
            {   # video
                "filepath": video.name,
                "data_type": "video",
                "name": "media",
            },
        ],
    }


def validate_h5(acq_h5: Path, dkey: str):
    """Print cam-frame and pulse counts so you can confirm channels 0/1."""
    try:
        import h5py
        import numpy as np
    except Exception as e:  # noqa
        print(f"  [validate] skipped (h5py/numpy not available: {e})")
        return
    try:
        with h5py.File(acq_h5, "r") as f:
            if dkey not in f:
                print(f"  [validate] '{dkey}' not in {acq_h5.name}; "
                      f"top-level keys: {list(f.keys())}")
                return
            try:
                sr = float(f["header/AcquisitionSampleRate"][()].ravel()[0])
            except Exception:
                sr = 250000.0
            raw = np.asarray(f[dkey][()]).ravel()
        cam = (raw >> CAM_CHANNEL) & 1
        pul = (raw >> PULSE_CHANNEL) & 1
        cam_rises = int(np.count_nonzero(np.diff(cam.astype(np.int8)) == 1))
        d = np.diff(pul.astype(np.int8))
        ps = np.where(d == 1)[0] / sr
        pe = np.where(d == -1)[0] / sr
        n = min(len(ps), len(pe))
        print(f"  [validate] {dkey}: {raw.size/sr:.1f}s @ {sr/1000:.0f} kHz | "
              f"cam frames (ch{CAM_CHANNEL} rises): {cam_rises} | "
              f"pulses (ch{PULSE_CHANNEL}): {n}")
        if n:
            widths = (pe[:n] - ps[:n]) * 1000.0
            print(f"             pulse width ~{np.median(widths):.0f} ms, "
                  f"first onset {ps[0]:.2f}s, last {ps[n-1]:.2f}s")
    except Exception as e:  # noqa
        print(f"  [validate] could not read {acq_h5.name}: {e}")


def save_json_safe(obj, path: Path):
    """Back up any existing file, then write; fall back to a numbered name if locked."""
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            try:
                shutil.copy2(path, bak)
                print(f"  backed up existing -> {bak.name}")
            except Exception as e:  # noqa
                print(f"  [!] could not back up {path.name}: {e}")
    text = json.dumps(obj, indent=2)
    try:
        path.write_text(text)
        print(f"  wrote {path.name}")
        return path
    except (PermissionError, OSError) as e:
        print(f"  [!] {path.name} not writable ({e}); trying a numbered name")
        for i in range(1, 100):
            alt = path.with_name(f"{path.stem}_{i}{path.suffix}")
            try:
                alt.write_text(text)
                print(f"  wrote {alt.name}")
                return alt
            except (PermissionError, OSError):
                continue
    print(f"  [!] could not write any JSON for {path.parent}")
    return None


def process(folder: Path, out_name: str, dry: bool, validate: bool):
    print(f"\n=== {folder} ===")
    if not folder.is_dir():
        print("  [!] not a folder; skipping")
        return
    try:
        video = find_video(folder)
        acq = find_acq_h5(folder)
        right = find_line_csv(folder, "right")
        left = find_line_csv(folder, "left")
    except FileNotFoundError as e:
        print(str(e)); print("  skipping this folder."); return

    dkey = sweep_key(acq.name)
    print(f"  video : {video.name}")
    print(f"  acq h5: {acq.name}  ->  {dkey}")
    print(f"  right : {right.name}  (green  {RIGHT_COLOR})")
    print(f"  left  : {left.name}  (magenta {LEFT_COLOR})")

    if validate:
        validate_h5(acq, dkey)

    obj = build_json(video, acq, right, left, dkey)
    out = folder / (out_name if out_name else f"{video.stem}_data1.json")
    if dry:
        print("  --dry-run: JSON that WOULD be written:")
        print("  " + json.dumps(obj, indent=2).replace("\n", "\n  "))
    else:
        save_json_safe(obj, out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="one session folder (overrides the default 3)")
    ap.add_argument("--dirs", nargs="+", help="several session folders")
    ap.add_argument("--out-name", default=None,
                    help="output JSON filename (default <video_stem>_data1.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written, write nothing")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip opening the h5 to count cam frames / pulses")
    args = ap.parse_args()

    if args.dir:
        folders = [args.dir]
    elif args.dirs:
        folders = args.dirs
    else:
        folders = DEFAULT_DIRS

    for d in folders:
        process(Path(d), args.out_name, args.dry_run, not args.no_validate)
    print("\nDone.")


if __name__ == "__main__":
    main()
