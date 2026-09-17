# Tests

Automated tests for `01_biomechanical_processing.py`.

The tests have two layers:

- **Unit tests** check individual functions against inputs with a known answer.
- **The regression test** runs whole pipeline on a small committed dataset and compares the output against a stored reference. 

## Running
From the repository root, in the analysis environment:

```bash
pytest -v
```

The same command runs in continuous integration (`.github/workflows/tests.yml`).

## Layout

| File | Purpose |
| --- | --- |
| `conftest.py` | Loads the processing script as the module `biomech`. (Needed because a module name cannot start with a digit.) |
| `test_biomechanics.py` | Unit tests for individual functions (calculations, column/metadata handling, QC metrics, input guards). |
| `test_regression.py` | End-to-end test: runs the pipeline on the fixed data and compares the summary to the reference output. |
| `compare_summary.py` | Column-by-column comparison of two `results_summary.csv` files, used by the regression test and runnable from the command line. |
| `build_fixture.py` | Regenerates the fixed data in `data/` from the full compiled CSVs. Run only when changing which samples are selected. |
| `data/` | The fixed data: five samples subset from the full dataset (~1.3 MB, gzipped). |
| `expected/results_summary.csv` | Reference output the regression test checks against. |

## What the unit tests cover

- **Calculations**: modulus of a linear stress-strain curve, hysteresis of a load/unload loop, stress relaxation of a known decay, and stress scaling with force.
- **Columns and metadata**: matching of truncated column names, date and replicate normalisation, and the FileName-based metadata join.
- **QC metrics**: SNR estimated on the preload baseline, its noise floor, and the rule that SNR gates only the failure segment.
- **Safety/guards**: bad inputs (zero cross-sectional area, short or missing segments) return NaN / not-OK flags rather than raising.

## The regression test and the reference output

`test_regression.py` runs the pipeline on `data/` and compares the resulting summary to `expected/results_summary.csv` within a numerical tolerance (plot-path columns are ignored; NaN matches NaN). A change to any reported value will fail the test.
Useful to to assess impact of changes to analysis approach/script and make sure additional modifications don't change the data/output in an unexpected or unintended way.

```bash
python 01_biomechanical_processing.py \
    --raw-dir tests/data --results-dir tests/expected --no-plots \
    --force-file compiled_Force_N_phase_140526_1.csv.gz \
    --disp-file  compiled_Displacement_mm_phase_140526_1.csv.gz \
    --size-file  compiled_Size_mm_phase_140526_1.csv.gz
```

If the fixed data itself needs to change, run `python tests/build_fixture.py` first (requires the full compiled CSVs in `raw_data/`).
The regression tests skip themselves if the fixed data or reference output is absent, so the other checks still run without them.
