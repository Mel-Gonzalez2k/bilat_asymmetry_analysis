#!/usr/bin/env python3
r"""
Keep only the rows whose Frame is within a range (inclusive); drop the rest.

Writes a NEW file next to the original (originals are never modified):
    left_whisker2.csv  ->  left_whisker2_frames_60135_63384.csv

Works on .csv or .xlsx and preserves the quoted multi-value X/Y point cells.

Usage
-----
    uv run trim_frames.py --file "E:\...\left_whisker2.csv"
    uv run trim_frames.py --file "...\left_whisker2.csv" --lo 60135 --hi 63384
    uv run trim_frames.py --file "...\right_whisker1.csv"          # same default range
    uv run trim_frames.py --file "...\left_whisker2.csv" --inplace # overwrite original

Dependencies: pandas (+ openpyxl only if the file is .xlsx).
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["pandas", "openpyxl"]
# ///

import argparse
from pathlib import Path
import pandas as pd

LO_DEFAULT, HI_DEFAULT = 60135, 63384
FRAME_COL = "Frame"


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def save(df: pd.DataFrame, path: Path):
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def save_safe(df, path: Path):
    try:
        save(df, path); print(f"wrote {path}"); return path
    except (PermissionError, OSError) as e:
        print(f"[!] {path.name} not writable ({e}); trying a numbered name")
        for i in range(1, 100):
            alt = path.with_name(f"{path.stem}_{i}{path.suffix}")
            try:
                save(df, alt); print(f"wrote {alt}"); return alt
            except (PermissionError, OSError):
                continue
    print(f"[!] could not write output for {path}"); return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the whisker CSV/XLSX to trim")
    ap.add_argument("--lo", type=int, default=LO_DEFAULT, help="lowest Frame to keep")
    ap.add_argument("--hi", type=int, default=HI_DEFAULT, help="highest Frame to keep")
    ap.add_argument("--col", default=FRAME_COL, help="name of the frame column")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the original instead of writing a new file")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    df = load(path)
    if args.col not in df.columns:
        raise SystemExit(f"No '{args.col}' column. Columns found: {list(df.columns)}")

    frame = pd.to_numeric(df[args.col], errors="coerce")
    keep = (frame >= args.lo) & (frame <= args.hi)
    out = df[keep].copy()

    print(f"{path.name}: {len(df)} rows -> kept {len(out)} "
          f"(Frame {args.lo}-{args.hi} inclusive), dropped {len(df) - len(out)}")
    if len(out):
        k = pd.to_numeric(out[args.col])
        print(f"  kept Frame range: {int(k.min())} .. {int(k.max())}")
    else:
        print("  [!] 0 rows matched — check the range or the Frame column.")

    dest = path if args.inplace else path.with_name(
        f"{path.stem}_frames_{args.lo}_{args.hi}{path.suffix}")
    save_safe(out, dest)


if __name__ == "__main__":
    main()
