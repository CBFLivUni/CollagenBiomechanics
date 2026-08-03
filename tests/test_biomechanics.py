"""
Unit tests for 01_biomechanical_processing.py

Grouped by intent:
  1. Analytic ground-truth   -- synthetic signals with a known answer
  2. Parsing / bookkeeping    -- the string munging where silent corruption hides
  3. QC metrics               -- do the filters behave as claimed
  4. Guards / edge cases      -- degenerate input returns NaN/None, never raises

Run:  pytest -v
"""
import numpy as np
import pandas as pd
import pytest

import biomech  # provided by conftest.py


# =====================================================================
# 1. ANALYTIC GROUND TRUTH
#    Build inputs where the correct output can be worked out by hand,
#    then assert the function reproduces it within tolerance.
# =====================================================================

def test_modulus_of_perfectly_linear_ramp():
    """A linear stress-strain curve of known slope E must yield max_mod == E."""
    E = 500.0            # target modulus (MPa)
    CSA = 0.02           # mm^2
    length = 5.0         # mm
    n = 400

    disp = np.linspace(0, 0.1, n)                 # mm  -> strain 0..0.02
    strain = disp / length
    stress = E * strain                           # exactly linear
    force = stress * CSA
    # add a tiny post-peak drop so there is a defined pre-peak region
    force = np.concatenate([force, force[::-1][:50]])
    disp = np.concatenate([disp, disp[-1] + np.linspace(0, 0.01, 50)])

    fail_df = pd.DataFrame({"Force_N": force, "Displacement_mm": disp})
    mod = biomech.compute_sliding_window_modulus(
        fail_df, CSA_true=CSA, sample_length=length
    )
    assert mod["global_idx"] is not None
    assert mod["modulus"] == pytest.approx(E, rel=1e-3)
    # a perfectly linear window should fit essentially exactly
    assert mod["r2"] == pytest.approx(1.0, abs=1e-6)
    assert mod["n_points"] >= biomech.MODULUS_MIN_POINTS_IN_WINDOW


def test_hysteresis_zero_for_identical_load_unload():
    """If unloading retraces the loading path, dissipated energy is ~0."""
    # analyze_preconditioning expects a df with the instrument columns.
    n = 300
    stretch_disp = np.linspace(0, 0.1, n)
    force = 10.0 * stretch_disp          # linear elastic
    recover_disp = stretch_disp[::-1]
    recover_force = 10.0 * recover_disp  # identical path back

    df = pd.DataFrame({
        "SetName": ["5x pre-conditioning"] * (2 * n),
        "Cycle": ["1-Stretch"] * n + ["1-Recover"] * n,
        "Time_S": np.arange(2 * n, dtype=float),
        "Size_mm": [5.0] * (2 * n),
        "Displacement_mm": np.concatenate([stretch_disp, recover_disp]),
        "Force_N": np.concatenate([force, recover_force]),
    })
    res = biomech.analyze_preconditioning(df, CSA_true=0.02)
    assert res["_ok"]
    # a closed identical loop encloses no area
    assert abs(res["hyst_pct"]) < 1.0   # percent


def test_hysteresis_positive_for_lossy_loop():
    """Unloading below the loading curve must give positive dissipation."""
    n = 300
    stretch_disp = np.linspace(0, 0.1, n)
    load = 10.0 * stretch_disp
    recover_disp = stretch_disp[::-1]
    unload = 6.0 * recover_disp          # sits below the loading curve
    df = pd.DataFrame({
        "SetName": ["5x pre-conditioning"] * (2 * n),
        "Cycle": ["1-Stretch"] * n + ["1-Recover"] * n,
        "Time_S": np.arange(2 * n, dtype=float),
        "Size_mm": [5.0] * (2 * n),
        "Displacement_mm": np.concatenate([stretch_disp, recover_disp]),
        "Force_N": np.concatenate([load, unload]),
    })
    res = biomech.analyze_preconditioning(df, CSA_true=0.02)
    assert res["_ok"]
    assert res["hyst_pct"] > 0
    assert res["hyst_energy"] > 0


def test_stress_relaxation_exponential_decay():
    """
    Synthetic hold: stress decays from s_peak to a known s60.
    stress_relax_60 (%) should equal (s_peak - s60)/s_peak * 100.
    """
    CSA = 0.02
    fs = 100                             # samples/s
    t = np.arange(0, 65, 1 / fs)         # 65 s at 100 Hz
    s_peak_stress = 10.0                 # MPa
    s60_stress = 6.0                     # MPa target at t=60
    tau = 60.0 / np.log(s_peak_stress / s60_stress)
    stress = s_peak_stress * np.exp(-t / tau)
    force = stress * CSA                 # relax_block Force column

    # analyze_hold wants a Preload cycle (baseline) + a Hold cycle
    preload_n = 50
    df = pd.DataFrame({
        "SetName": ["Stress-relax"] * (preload_n + len(t)),
        "Cycle": ["1-Preload"] * preload_n + ["1-Hold"] * len(t),
        "Time_S": np.concatenate([np.linspace(-0.5, 0, preload_n), t]),
        "Size_mm": [5.0] * (preload_n + len(t)),
        "Displacement_mm": [0.0] * (preload_n + len(t)),
        "Force_N": np.concatenate([np.zeros(preload_n), force]),
    })
    res = biomech.analyze_hold(df, CSA_true=CSA)
    expected_pct = (s_peak_stress - s60_stress) / s_peak_stress * 100
    assert res["stress_relax_60"] == pytest.approx(expected_pct, abs=1.0)


def test_stress_scales_linearly_with_force():
    """Doubling the whole force trace must double failure stress (property test)."""
    n = 200
    disp = np.linspace(0, 0.2, n)
    base = pd.DataFrame({
        "SetName": ["fail"] * n, "Cycle": ["1-Stretch"] * n,
        "Time_S": np.arange(n, dtype=float), "Size_mm": [5.0] * n,
        "Displacement_mm": disp, "Force_N": np.linspace(0, 1.0, n),
    })
    doubled = base.copy()
    doubled["Force_N"] = base["Force_N"] * 2
    r1 = biomech.analyze_failure(base, CSA_true=0.02, sample_length=5.0)
    r2 = biomech.analyze_failure(doubled, CSA_true=0.02, sample_length=5.0)
    assert r2["failure_stress"] == pytest.approx(2 * r1["failure_stress"], rel=1e-6)


# =====================================================================
# 2. PARSING / BOOKKEEPING
#    Truncated-suffix matching and metadata normalisation are where a
#    silent wrong-sample join would come from.
# =====================================================================

@pytest.mark.parametrize("remainder,expected,ok", [
    ("SetName", "SetName", True),
    ("SetNam", "SetName", True),     # Excel 31-char truncation
    ("SetN",   "SetName", True),
    ("Force_",  "Force_N", True),
    ("",       "SetName", False),    # empty never matches
    ("Xyz",    "SetName", False),
])
def test_suffix_matches(remainder, expected, ok):
    assert biomech._suffix_matches(remainder, expected) is ok


def test_find_col_flags_ambiguous_truncation():
    """'S' could be SetName or Size_mm -> must refuse to guess (return None)."""
    df = pd.DataFrame(columns=["Time_S", "smpl_S"])
    assert biomech.find_col(df, "smpl", "SetName") is None


def test_find_col_resolves_clean_truncation():
    df = pd.DataFrame(columns=["Time_S", "smpl_SetNam", "smpl_Force_"])
    assert biomech.find_col(df, "smpl", "SetName") == "smpl_SetNam"
    assert biomech.find_col(df, "smpl", "Force_N") == "smpl_Force_"


@pytest.mark.parametrize("raw,out", [
    (210330, "210330"),
    ("210330", "210330"),
    (330, "000330"),          # zero-pad to 6 -- guards the int-inference join bug
    (np.int64(210330), "210330"),
])
def test_normalise_date_id(raw, out):
    assert biomech._normalise_date_id(raw) == out


@pytest.mark.parametrize("raw,out", [
    (1.0, "1"), (1, "1"), ("1", "1"), (10.0, "10"),
])
def test_normalise_replicate(raw, out):
    assert biomech._normalise_replicate(raw) == out


def test_lookup_metadata_prefers_exact_filename():
    """C3 and C3.2 must NOT collide when FileName is available."""
    meta = pd.DataFrame({
        "FileName": ["210330 MRC Sample C3Data.csv", "210330 MRC Sample C3.2Data.csv"],
        "Date_ID": ["210330", "210330"],
        "Sample_ID": ["C", "C"],
        "Replicate": ["3", "3"],
    })
    rows, how = biomech.lookup_metadata(meta, "210330 MRC Sample C3.2Data")
    assert how == "FileName"
    assert len(rows) == 1
    assert rows.iloc[0]["FileName"] == "210330 MRC Sample C3.2Data.csv"


# =====================================================================
# 3. QC METRICS
# =====================================================================

def test_snr_high_for_clean_ramp():
    """Clean ramp above a quiet baseline -> high SNR."""
    y = np.linspace(0, 10, 500)
    assert biomech.compute_snr(y) > biomech.SNR_THRESHOLD


def test_snr_low_for_pure_noise():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)            # no real signal
    snr = biomech.compute_snr(y)
    assert (not np.isfinite(snr)) or snr < biomech.SNR_THRESHOLD


def test_snr_uses_preload_cycle_as_baseline():
    """
    Noise must come from the 1-Preload phase, not the whole segment. Build a
    quiet preload of known SD plus a clean ramp to a known peak, and check
    SNR ~ peak / preload_noise (i.e. the loading gradient is NOT counted as noise).
    """
    rng = np.random.default_rng(1)
    n_pre, n_load = 200, 400
    preload = rng.normal(0.0, 0.02, n_pre)          # noise SD ~0.02 N
    ramp = np.linspace(0.0, 5.0, n_load)            # peak ~5 N above baseline
    force = np.concatenate([preload, ramp])
    cycle = ["1-Preload"] * n_pre + ["1-Stretch"] * n_load
    snr = biomech.compute_snr(force, cycle)
    assert np.isfinite(snr)
    assert 150 < snr < 400                          # ~5 / 0.02


def test_snr_noise_floor_prevents_blowup():
    """A perfectly flat (quantisation-limited) preload must stay finite, not inf."""
    force = np.concatenate([np.full(200, 0.10), np.linspace(0.10, 0.40, 400)])
    cycle = ["1-Preload"] * 200 + ["1-Stretch"] * 400
    snr = biomech.compute_snr(force, cycle)
    assert np.isfinite(snr) and snr < 1e4           # floored at q/sqrt(12)


def test_snr_nan_for_flat_signal():
    """No rise above baseline -> no signal -> NaN (cannot pass QC)."""
    assert not np.isfinite(biomech.compute_snr(np.full(500, 2.0)))


def test_snr_gates_failure_but_not_gentle_segments():
    """
    The C2 case: a well-shaped segment (passes residual + range) whose peak
    does not clear a high preload, so its peak-over-preload SNR is low/undefined.
    It must FAIL when SNR gates (failure segment) but PASS when SNR does not gate
    (preconditioning / hold), so a good specimen is not excluded on a metric that
    is only well-posed for the failure curve.
    """
    rng = np.random.default_rng(3)
    n_pre, n_load = 200, 600
    preload = np.full(n_pre, 0.20) + rng.normal(0, 3e-4, n_pre)   # high, quiet preload
    load = 0.20 - 0.05 * np.sin(np.linspace(0, np.pi, n_load))    # smooth dip, never clears preload
    force = np.concatenate([preload, load])
    cycle = ["1-Preload"] * n_pre + ["1-Stretch"] * n_load

    gated = biomech.segment_qc(force, cycle, "failure", gate_snr=True)
    ungated = biomech.segment_qc(force, cycle, "pre-conditioning", gate_snr=False)

    # precondition: it's a clean, ranged segment that simply has poor SNR
    assert gated["snr_pass"] is False
    assert gated["resid_pass"] is True and gated["range_pass"] is True
    # the gate is the only thing that differs
    assert gated["pass"] is False
    assert ungated["pass"] is True


# =====================================================================
# 4. GUARDS / EDGE CASES  -- must degrade gracefully, never raise
# =====================================================================

def test_zero_csa_gives_nan_modulus():
    fail_df = pd.DataFrame({
        "Force_N": np.linspace(0, 1, 50),
        "Displacement_mm": np.linspace(0, 0.1, 50),
    })
    mod = biomech.compute_sliding_window_modulus(
        fail_df, CSA_true=0.0, sample_length=5.0)
    assert np.isnan(mod["modulus"]) and mod["global_idx"] is None
    assert mod["n_points"] == 0


def test_savgol_safe_passes_through_short_segment():
    y = np.array([1.0, 2.0, 3.0])        # shorter than min window
    out = biomech.apply_savgol_safe(y, 101, 3)
    np.testing.assert_array_equal(out, y)


def test_missing_preconditioning_segment_returns_not_ok():
    df = pd.DataFrame({
        "SetName": ["fail"] * 10, "Cycle": ["1-Stretch"] * 10,
        "Time_S": np.arange(10.0), "Size_mm": [5.0] * 10,
        "Displacement_mm": np.linspace(0, 1, 10), "Force_N": np.linspace(0, 1, 10),
    })
    res = biomech.analyze_preconditioning(df, CSA_true=0.02)
    assert res["_ok"] is False
