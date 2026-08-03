"""
Build a small, self-contained regression fixture under tests/data/ by
subsetting the compiled CSVs to a handful of samples. Pure-pandas version of
make_test_data.sh (no csvkit dependency), so it runs anywhere pytest does.

Chosen samples deliberately span the interesting cases:
  A1     -- everything passes
  B10    -- hold fails QC (residual)
  C3     -- collides with C3.2 on the (Date,Sample,Replicate) triple
  C3.2   -- the collision partner; exercises exact-FileName metadata matching
  C6     -- lowest hold SNR in the set

Run from the repo root:  python tests/build_fixture.py
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)

FORCE = "compiled_Force_N_phase_140526_1.csv"
DISP = "compiled_Displacement_mm_phase_140526_1.csv"
SIZE = "compiled_Size_mm_phase_140526_1.csv"
META = "compiled_metadata_with_filenames.csv"

# Sample bases to keep (as they appear before the _<suffix>).
WANTED = [
    "210330 MRC Sample A1Data",
    "210330 MRC Sample B10Data",
    "210330 MRC Sample C3Data",
    "210330 MRC Sample C3.2Data",
    "210330 MRC Sample C6Data",
]


def subset_wide(infile: Path, outfile: Path):
    df = pd.read_csv(infile, low_memory=False, encoding="utf-8-sig")
    keep = ["Time_S"]
    for col in df.columns:
        if col == "Time_S":
            continue
        for base in WANTED:
            # match "<base>_<suffix>" exactly (underscore boundary)
            if col.startswith(base + "_"):
                keep.append(col)
                break
    df[keep].to_csv(outfile, index=False)
    print(f"  {outfile.name}: {len(keep)} cols, {len(df)} rows")


def subset_meta(infile: Path, outfile: Path):
    df = pd.read_csv(infile, encoding="utf-8-sig")
    targets = {b + ".csv" for b in WANTED}
    sub = df[df["FileName"].astype(str).str.strip().isin(targets)]
    sub.to_csv(outfile, index=False)
    print(f"  {outfile.name}: {len(sub)} rows")


if __name__ == "__main__":
    print("Writing fixture to tests/data/ ...")
    subset_wide(ROOT / FORCE, OUT / FORCE)
    subset_wide(ROOT / DISP, OUT / DISP)
    subset_wide(ROOT / SIZE, OUT / SIZE)
    subset_meta(ROOT / META, OUT / META)
    print("Done.")
