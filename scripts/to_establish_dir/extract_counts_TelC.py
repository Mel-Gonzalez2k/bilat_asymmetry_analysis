#!/usr/bin/env python3
r"""
Report camera-TTL time points, pulse-pal intervals, and video frame counts
for every (.h5, .mp4) recording pair, across a whole experiment tree.

Works on both dataset layouts:

  opto:  MRN_opto12_..._5x_2026-05-15_001_0001.h5  +  ..._5x0_noB.mp4
  TeLC:  WA021_L_MRN_TeLC_01_pre_TeLC_2026-03-22_0001.h5
                                       +  WA021_L_MRN_TeLC_01_pre_TeLC0_noB_rot180.mp4

Pairing is done by stripping a flexible tail from each filename and matching
on the shared stem, so the digit before "_noB", an added "_rot180", and the
"_001" block in the opto names are all handled automatically. .csv and
data.json files are ignored.

The .h5 stores one bit-packed digital dataset at "sweep_0001/digitalScans":
bit 0 = camera TTL (rising edges -> time points), bit 1 = pulse-pal TTL
(high intervals). These mirror the channel/transition fields in data.json.

Directory layout (TeLC example):
    TeLC_Silencing\<subject>\<date>\<session>\  (the .h5 and .mp4 live here)
e.g. ...\TeLC_Silencing\WA020\20260322\Pre_Day0

Usage
-----
One session folder (no recursion):
    uv run extract_counts.py "E:\...\TeLC_Silencing\WA020\20260322\Pre_Day0"

Whole experiment (recurse into every subject/date/session):
    uv run extract_counts.py "E:\...\TeLC_Silencing" --recursive

Options:
    --recursive / -r     Walk all subfolders instead of just the given one.
    --csv counts.csv     Write a summary table across everything processed.
    --data-key <path>    HDF5 dataset path (default sweep_0001/digitalScans).
    --cam-bit N          Bit index of camera TTL (default 0).
    --pulse-bit N        Bit index of pulse-pal TTL (default 1).

Dependencies: h5py, numpy, opencv-python (installed automatically by uv via
the inline block below).
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["h5py", "numpy", "opencv-python"]
# ///

import argparse
import csv
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

DEFAULT_KEY = "sweep_0001/digitalScans"

# Minimum stem similarity to accept an h5<->mp4 pair when stems aren't identical
# (covers small naming inconsistencies like an extra "_F_"/"_M_" or "_2.0" that
# appear on only one of the two files). Genuine mismatches score far below this.
SIM_THRESHOLD = 0.80

# Flexible filename tails. Anything before the matched tail is the shared stem.
#   h5  : _<YYYY-MM-DD>_<digits...>.h5   (covers _2026-03-22_0001.h5 and
#                                         _2026-05-15_001_0001.h5)
#   mp4 : <optional SINGLE rep digit>_noB<optional _rot###>.mp4
# NOTE: the mp4 rep digit is a single "\d?" -- using "\d*" would greedily eat
# the trailing digit of names like "..._Day7" or "..._Day10", breaking pairing.
H5_TAIL = re.compile(r"_\d{4}-\d{2}-\d{2}_[0-9_]*\d\.h5$")
MP4_TAIL = re.compile(r"\d?_noB(?:_rot\d+)?\.mp4$")


def h5_stem(name: str):
    m = H5_TAIL.search(name)
    return name[: m.start()] if m else None


def mp4_stem(name: str):
    m = MP4_TAIL.search(name)
    return name[: m.start()] if m else None


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def pair_files(h5_by: dict, mp4_by: dict):
    """Return (pairs, lonely_h5, lonely_mp4).

    pairs is a list of (h5_stem, h5_name, mp4_name, note) where note is "" for
    an exact stem match or "~sim=0.97" for an approximate one. Strategy:
      1. Match identical stems first.
      2. For whatever h5/mp4 remain, greedily match highest-similarity pairs
         above SIM_THRESHOLD. This recovers files that differ only by a stray
         "_F_"/"_M_"/"_2.0" on one side.
    """
    pairs = []
    h5_left = dict(h5_by)   # stem -> name
    mp4_left = dict(mp4_by)

    # 1) exact-stem matches
    for stem in sorted(set(h5_left) & set(mp4_left)):
        pairs.append((stem, h5_left[stem], mp4_left[stem], ""))
        del h5_left[stem]
        del mp4_left[stem]

    # 2) fuzzy matches on the remainder
    while h5_left and mp4_left:
        best = None  # (sim, h5_stem, mp4_stem)
        for hs in h5_left:
            for ms in mp4_left:
                sim = similarity(hs, ms)
                if best is None or sim > best[0]:
                    best = (sim, hs, ms)
        if best is None or best[0] < SIM_THRESHOLD:
            break
        _, hs, ms = best
        pairs.append((hs, h5_left[hs], mp4_left[ms], f"~sim={best[0]:.2f}"))
        del h5_left[hs]
        del mp4_left[ms]

    return pairs, h5_left, mp4_left


def count_rising_edges(bit_trace: np.ndarray) -> int:
    return int(np.count_nonzero((bit_trace[1:] == 1) & (bit_trace[:-1] == 0)))


def count_intervals(bit_trace: np.ndarray) -> int:
    d = np.diff(bit_trace.astype(np.int8))
    rises = int(np.count_nonzero(d == 1))
    if bit_trace.size and bit_trace[0] == 1:
        rises += 1
    return rises


def read_h5_counts(h5_path: Path, data_key: str, cam_bit: int, pulse_bit: int):
    import h5py
    with h5py.File(h5_path, "r") as f:
        if data_key not in f:
            raise KeyError(
                f"dataset '{data_key}' not found in {h5_path.name}. "
                f"Top-level keys: {list(f.keys())}"
            )
        raw = f[data_key][()]
    raw = np.asarray(raw).ravel()
    cam = ((raw >> cam_bit) & 1).astype(np.int8)
    pulse = ((raw >> pulse_bit) & 1).astype(np.int8)
    return count_rising_edges(cam), count_intervals(pulse), raw.size


def read_video_frames(mp4_path: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise IOError(f"could not open video {mp4_path.name}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def find_session_dirs(root: Path, recursive: bool):
    """Yield folders that contain at least one .h5 file."""
    if not recursive:
        yield root
        return
    seen = set()
    for h5 in sorted(root.rglob("*.h5")):
        d = h5.parent
        if d not in seen:
            seen.add(d)
            yield d


def path_labels(session_dir: Path, root: Path):
    """Best-effort subject/date/session from the folder path.
    For ...\\TeLC_Silencing\\WA020\\20260322\\Pre_Day0 returns
    ('WA020','20260322','Pre_Day0'). Falls back gracefully."""
    try:
        rel = session_dir.relative_to(root)
        parts = rel.parts if rel.parts else session_dir.parts
    except ValueError:
        parts = session_dir.parts
    subj = date = sess = ""
    if len(parts) >= 3:
        subj, date, sess = parts[-3], parts[-2], parts[-1]
    elif len(parts) == 2:
        subj, sess = parts[-2], parts[-1]
    elif len(parts) == 1:
        sess = parts[-1]
    return subj, date, sess


def process_dir(session_dir: Path, root: Path, args, rows):
    names = [p.name for p in session_dir.iterdir() if p.is_file()]
    h5_by = {s: n for n in names if (s := h5_stem(n)) is not None}
    mp4_by = {s: n for n in names if (s := mp4_stem(n)) is not None}

    if not (h5_by or mp4_by):
        return  # nothing recording-like here

    pairs, lonely_h5, lonely_mp4 = pair_files(h5_by, mp4_by)

    subj, date, sess = path_labels(session_dir, root)
    header = " / ".join(x for x in (subj, date, sess) if x) or str(session_dir)
    print(f"\n########## {header} ##########")
    print(f"({session_dir})")

    for s in sorted(lonely_h5):
        print(f"  [warn] h5 with no matching mp4: {lonely_h5[s]}")
    for s in sorted(lonely_mp4):
        print(f"  [warn] mp4 with no matching h5: {lonely_mp4[s]}")

    for s, h5_name, mp4_name, note in pairs:
        tag = f"  ({note} fuzzy match)" if note else ""
        print(f"\n  === {s} ==={tag}")
        try:
            n_cam, n_pulse, n_samp = read_h5_counts(
                session_dir / h5_name, args.data_key, args.cam_bit, args.pulse_bit)
        except Exception as e:
            print(f"    [error reading h5] {e}")
            continue
        try:
            n_frames = read_video_frames(session_dir / mp4_name)
        except Exception as e:
            print(f"    [error reading mp4] {e}")
            n_frames = -1

        diff = (n_frames - n_cam) if n_frames >= 0 else None
        print(f"    h5  : {h5_name}")
        print(f"          samples in dataset : {n_samp:,}")
        print(f"          camera TTL events (time points) : {n_cam:,}")
        print(f"          pulse-pal intervals             : {n_pulse}")
        print(f"    mp4 : {mp4_name}")
        print(f"          video frames                    : {n_frames:,}")
        if diff is not None:
            flag = "  <-- MISMATCH" if diff != 0 else "  (match)"
            print(f"          frames - cam events             : {diff:+d}{flag}")

        rows.append({
            "subject": subj, "date": date, "session": sess,
            "stem": s, "h5": h5_name, "mp4": mp4_name,
            "n_samples": n_samp, "cam_ttl_events": n_cam,
            "pulse_pal_intervals": n_pulse, "video_frames": n_frames,
            "frames_minus_cam": diff, "folder": str(session_dir),
        })


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory",
                    help="A session folder, or an experiment root with --recursive.")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Walk all subfolders; process every folder containing .h5 files.")
    ap.add_argument("--data-key", default=DEFAULT_KEY)
    ap.add_argument("--cam-bit", type=int, default=0)
    ap.add_argument("--pulse-bit", type=int, default=1)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    rows = []
    for session_dir in find_session_dirs(root, args.recursive):
        process_dir(session_dir, root, args, rows)

    n_folders = len({r["folder"] for r in rows})
    print(f"\n==================== Processed {len(rows)} pair(s) "
          f"across {n_folders} folder(s). ====================")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote summary: {args.csv}")


if __name__ == "__main__":
    main()