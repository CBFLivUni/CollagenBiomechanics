"""
Compare two results_summary.csv files for the regression test.

Rules:
  * rows are matched on the 'File name' key (order-independent)
  * plot-path columns are ignored (they are absolute paths, not results)
  * numeric columns compared with a relative+absolute tolerance, NaN==NaN
  * bool / string columns compared exactly

Usable as a library (compare_summaries -> list of human-readable diffs) or
from the command line for CI:

    python tests/compare_summary.py got.csv expected.csv --rtol 1e-6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = "File name"
IGNORE_COLS = {"Force-Time plot", "Force-Extension plot"}


def compare_summaries(got: pd.DataFrame, expected: pd.DataFrame,
                      rtol: float = 1e-6, atol: float = 1e-9) -> list[str]:
    diffs: list[str] = []

    if KEY not in got.columns or KEY not in expected.columns:
        return [f"Missing '{KEY}' column in one of the frames."]

    got = got.set_index(KEY).sort_index()
    expected = expected.set_index(KEY).sort_index()

    # Row keys
    missing = set(expected.index) - set(got.index)
    extra = set(got.index) - set(expected.index)
    if missing:
        diffs.append(f"Rows missing from output: {sorted(missing)}")
    if extra:
        diffs.append(f"Unexpected extra rows: {sorted(extra)}")
    shared_rows = expected.index.intersection(got.index)

    # Columns
    missing_cols = set(expected.columns) - set(got.columns) - IGNORE_COLS
    extra_cols = set(got.columns) - set(expected.columns) - IGNORE_COLS
    if missing_cols:
        diffs.append(f"Columns missing from output: {sorted(missing_cols)}")
    if extra_cols:
        diffs.append(f"Unexpected extra columns: {sorted(extra_cols)}")
    shared_cols = [c for c in expected.columns
                   if c in got.columns and c not in IGNORE_COLS]

    g = got.loc[shared_rows, shared_cols]
    e = expected.loc[shared_rows, shared_cols]

    for col in shared_cols:
        gc, ec = g[col], e[col]
        if pd.api.types.is_numeric_dtype(ec) and not pd.api.types.is_bool_dtype(ec):
            gv = pd.to_numeric(gc, errors="coerce").to_numpy(float)
            ev = pd.to_numeric(ec, errors="coerce").to_numpy(float)
            close = np.isclose(gv, ev, rtol=rtol, atol=atol, equal_nan=True)
            for key, ok, a, b in zip(shared_rows, close, gv, ev):
                if not ok:
                    diffs.append(f"[{col}] {key}: got {a!r}, expected {b!r}")
        else:
            for key in shared_rows:
                a, b = gc.loc[key], ec.loc[key]
                if pd.isna(a) and pd.isna(b):
                    continue
                if a != b:
                    diffs.append(f"[{col}] {key}: got {a!r}, expected {b!r}")
    return diffs


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diff two results_summary.csv files.")
    p.add_argument("got", type=Path)
    p.add_argument("expected", type=Path)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-9)
    a = p.parse_args(argv)

    diffs = compare_summaries(pd.read_csv(a.got), pd.read_csv(a.expected),
                              rtol=a.rtol, atol=a.atol)
    if diffs:
        print(f"REGRESSION: {len(diffs)} difference(s):")
        for d in diffs[:50]:
            print("  " + d)
        return 1
    print("OK: outputs match within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
