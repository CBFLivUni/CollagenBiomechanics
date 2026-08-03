"""
End-to-end regression test.

Runs the pipeline on the committed fixture (tests/data/) and compares the
per-tendon summary against the stored reference output (tests/expected/).

Regenerate the reference output after an intended change to the analysis:

    python tests/build_fixture.py            # only if the fixture changed
    python 01_biomechanical_processing.py \
        --raw-dir tests/data --results-dir tests/expected --no-plots \
        --force-file compiled_Force_N_phase_140526_1.csv.gz \
        --disp-file  compiled_Displacement_mm_phase_140526_1.csv.gz \
        --size-file  compiled_Size_mm_phase_140526_1.csv.gz

Review the diff on tests/expected/results_summary.csv before committing.
"""
from pathlib import Path

import pandas as pd
import pytest

import biomech  # from conftest.py
from compare_summary import compare_summaries  # tests/ is on sys.path under pytest

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXPECTED = HERE / "expected" / "results_summary.csv"
FIXTURE = DATA / "compiled_Force_N_phase_140526_1.csv.gz"

needs_fixture = pytest.mark.skipif(
    not (FIXTURE.exists() and EXPECTED.exists()),
    reason="fixture or reference output absent; see tests/build_fixture.py",
)

CLI = [
    "--no-plots",
    "--force-file", "compiled_Force_N_phase_140526_1.csv.gz",
    "--disp-file", "compiled_Displacement_mm_phase_140526_1.csv.gz",
    "--size-file", "compiled_Size_mm_phase_140526_1.csv.gz",
    "--meta-file", "compiled_metadata_with_filenames.csv",
]


@needs_fixture
def test_pipeline_matches_reference_output(tmp_path):
    rc = biomech.main(["--raw-dir", str(DATA), "--results-dir", str(tmp_path)] + CLI)
    assert rc == 0

    got = pd.read_csv(tmp_path / "results_summary.csv")
    expected = pd.read_csv(EXPECTED)

    diffs = compare_summaries(got, expected, rtol=1e-6, atol=1e-9)
    assert not diffs, "summary differs from reference:\n" + "\n".join(diffs[:50])


@needs_fixture
def test_c3_and_c32_resolve_to_separate_recordings(tmp_path):
    # The fixture contains both C3 and C3.2, which share the same
    # (Date, Sample, Replicate) triple and identical metadata. The FileName
    # join must still map each to its own recording; a collapsed join would
    # give them the same data-derived values.
    rc = biomech.main(["--raw-dir", str(DATA), "--results-dir", str(tmp_path)] + CLI)
    assert rc == 0

    got = pd.read_csv(tmp_path / "results_summary.csv")
    names = set(got["File name"])
    assert "210330 MRC Sample C3Data.csv" in names
    assert "210330 MRC Sample C3.2Data.csv" in names

    force_col = "Failure force (N) - corrected"
    c3 = got.loc[got["File name"] == "210330 MRC Sample C3Data.csv", force_col].iloc[0]
    c32 = got.loc[got["File name"] == "210330 MRC Sample C3.2Data.csv", force_col].iloc[0]
    assert c3 != c32
