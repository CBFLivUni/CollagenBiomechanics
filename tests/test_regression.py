"""
End-to-end regression test.

Runs the whole pipeline on the committed fixture (tests/data/) and checks that
the per-tendon summary still matches the golden output (tests/expected/). This
is the test that lets a referee reproduce the numbers, and it fails loudly if a
refactor silently changes any reported parameter.

To refresh the golden output after an INTENTIONAL change to the method:
    python tests/build_fixture.py            # only if the fixture itself changed
    python 01_biomechanical_processing.py \
        --raw-dir tests/data --results-dir tests/expected --no-plots \
        --force-file compiled_Force_N_phase_140526_1.csv.gz \
        --disp-file  compiled_Displacement_mm_phase_140526_1.csv.gz \
        --size-file  compiled_Size_mm_phase_140526_1.csv.gz
    # then review the git diff on tests/expected/results_summary.csv before committing
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
    reason="fixture/golden output absent; run tests/build_fixture.py then regenerate",
)


@needs_fixture
def test_pipeline_matches_golden_output(tmp_path):
    rc = biomech.main([
        "--raw-dir", str(DATA),
        "--results-dir", str(tmp_path),
        "--no-plots",
        "--force-file", "compiled_Force_N_phase_140526_1.csv.gz",
        "--disp-file", "compiled_Displacement_mm_phase_140526_1.csv.gz",
        "--size-file", "compiled_Size_mm_phase_140526_1.csv.gz",
        "--meta-file", "compiled_metadata_with_filenames.csv",
    ])
    assert rc == 0, "pipeline returned non-zero"

    got = pd.read_csv(tmp_path / "results_summary.csv")
    expected = pd.read_csv(EXPECTED)

    diffs = compare_summaries(got, expected, rtol=1e-6, atol=1e-9)
    assert not diffs, "regression vs golden output:\n" + "\n".join(diffs[:50])


@needs_fixture
def test_c3_and_c32_do_not_collide(tmp_path):
    """
    The fixture deliberately contains both C3 and C3.2. Guards the metadata
    join: each must resolve to its OWN row (this is what the FileName-first
    lookup buys us over the (Date,Sample,Replicate) regex).
    """
    rc = biomech.main([
        "--raw-dir", str(DATA),
        "--results-dir", str(tmp_path),
        "--no-plots",
        "--force-file", "compiled_Force_N_phase_140526_1.csv.gz",
        "--disp-file", "compiled_Displacement_mm_phase_140526_1.csv.gz",
        "--size-file", "compiled_Size_mm_phase_140526_1.csv.gz",
        "--meta-file", "compiled_metadata_with_filenames.csv",
    ])
    assert rc == 0
    got = pd.read_csv(tmp_path / "results_summary.csv")
    names = set(got["File name"])
    assert "210330 MRC Sample C3Data.csv" in names
    assert "210330 MRC Sample C3.2Data.csv" in names
    # C3 and C3.2 happen to share identical metadata (same CSA), so CSA can't
    # tell them apart. What must differ is the DATA-derived result: they are
    # two different recordings, so a collapsed/duplicated join would show up as
    # identical failure force here.
    c3 = got.loc[got["File name"] == "210330 MRC Sample C3Data.csv",
                 "Failure force (N) - corrected"].iloc[0]
    c32 = got.loc[got["File name"] == "210330 MRC Sample C3.2Data.csv",
                  "Failure force (N) - corrected"].iloc[0]
    assert c3 != c32
