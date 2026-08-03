"""
Unit tests for 01_biomechanical_processing.py

Groups:
  1. Ground-truth checks   - synthetic inputs with an analytically known result
  2. Column and metadata   - suffix matching and metadata normalisation
  3. QC metrics            - SNR, residual and range filters
  4. Guards                - degenerate input returns NaN/None rather than raising

Run: pytest -v
"""
import numpy as np
import pandas as pd
import pytest

import biomech  # provided by conftest.py


# Ground-truth checks 

def test_modulus_of_perfectly_linear_ramp():
    """Linear stress-strain curve of slope E; reported modulus should equal E."""
    E = 500.0            # target modulus (MPa)
    CSA = 0.02           # mm^2
    length = 5.0         # mm
    n = 400

    disp = np.linspace(0, 0.1, n)                 # mm -> strain 0..0.02
    strain = disp / length
    stress = E * strain
    force = stress * CSA
    # small post-peak drop so a pre-peak region is defined
    force = np.concatenate([force, force[::-1][:50]])
    disp = np.concatenate([disp, disp[-1] + np.linspace(0, 0.01, 50)])

    fail_df = pd.DataFrame({"Force_N": force, "Displacement_mm": disp})
    mod = biomech.compute_sliding_window_modulus(
        fail_df, CSA_true=CSA, sample_length=length
    )
    assert mod["global_idx"] is not None
    assert mod["modulus"] == pytest.approx(E, rel=1e-3)
    assert mod["r2"] == pytest.approx(1.0, abs=1e-6)
    assert mod["n_points"] >= biomech.MODULUS_MIN_POINTS_IN_WINDOW


def test_hysteresis_zero_for_identical_load_unload():
    """Unloading that retraces the loading path encloses no area."""
    n = 300
    stretch_disp = np.linspace(0, 0.1, n)
    force = 10.0 * stretch_disp
    recover_disp = stretch_disp[::-1]
    recover_force = 10.0 * recover_disp

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
    assert abs(res["hyst_pct"]) < 1.0   # percent


def test_hysteresis_positive_for_lossy_loop():
    """Unloading below the loading curve gives positive dissipation."""
    n = 300
    stretch_disp = np.linspace(0, 0.1, n)
    load = 10.0 * stretch_disp
    recover_disp = stretch_disp[::-1]
    unload = 6.0 * recover_disp          # below the loading curve
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
    """Stress decaying from s_peak to a known s60 gives (s_peak-s60)/s_peak * 100."""
    CSA = 0.02
    fs = 100                             # samples/s
    t = np.arange(0, 65, 1 / fs)         # 65 s at 100 Hz
    s_peak_stress = 10.0                 # MPa
    s60_stress = 6.0                     # MPa at t=60
    tau = 60.0 / np.log(s_peak_stress / s60_stress)
    stress = s_peak_stress * np.exp(-t / tau)
    force = stress * CSA

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
    """Doubling the force trace doubles the failure stress."""
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


# Column and metadata 

@pytest.mark.parametrize("remainder,expected,ok", [
    ("SetName", "SetName", True),
    ("SetNam", "SetName", True),     # Excel 31-char truncation
    ("SetN",   "SetName", True),
    ("Force_",  "Force_N", True),
    ("",       "SetName", False),
    ("Xyz",    "SetName", False),
])
def test_suffix_matches(remainder, expected, ok):
    assert biomech._suffix_matches(remainder, expected) is ok


def test_find_col_flags_ambiguous_truncation():
    """'S' matches both SetName and Size_mm, so find_col returns None."""
    df = pd.DataFrame(columns=["Time_S", "smpl_S"])
    assert biomech.find_col(df, "smpl", "SetName") is None


def test_find_col_resolves_clean_truncation():
    df = pd.DataFrame(columns=["Time_S", "smpl_SetNam", "smpl_Force_"])
    assert biomech.find_col(df, "smpl", "SetName") == "smpl_SetNam"
    assert biomech.find_col(df, "smpl", "Force_N") == "smpl_Force_"


@pytest.mark.parametrize("raw,out", [
    (210330, "210330"),
    ("210330", "210330"),
    (330, "000330"),          # zero-padded to six characters
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
    """FileName match distinguishes C3 from C3.2, which share the name triple."""
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


# QC metrics 

def test_snr_high_for_clean_ramp():
    """Clean ramp above a unnoisy baseline gives high SNR."""
    y = np.linspace(0, 10, 500)
    assert biomech.compute_snr(y) > biomech.SNR_THRESHOLD


def test_snr_low_for_pure_noise():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    snr = biomech.compute_snr(y)
    assert (not np.isfinite(snr)) or snr < biomech.SNR_THRESHOLD


def test_snr_uses_preload_cycle_as_baseline():
    """
    Noise is taken from the 1-Preload phase, not the whole segment. For a unnoisy preload plus a clean ramp, SNR is approximately peak / preload SD, so the loading gradient does not contribute to the noise estimate.
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
    """A steady preload barely varies, so its noise estimate is ~0; the floor keeps SNR finite instead of dividing by near-zero."""
    force = np.concatenate([np.full(200, 0.10), np.linspace(0.10, 0.40, 400)])
    cycle = ["1-Preload"] * 200 + ["1-Stretch"] * 400
    snr = biomech.compute_snr(force, cycle)
    assert np.isfinite(snr) and snr < 1e4           # bounded by the noise floor


def test_snr_nan_for_flat_signal():
    """No rise above baseline gives no signal, so SNR is NaN and cannot pass."""
    assert not np.isfinite(biomech.compute_snr(np.full(500, 2.0)))


def test_snr_gates_failure_but_not_gentle_segments():
    """
    A segment that passes the residual and range filters but has low SNR (its peak does not clear a high preload) fails only where SNR gates, i.e. the failure segment. 
    Preconditioning and hold do not gate on SNR.
    """
    rng = np.random.default_rng(3)
    n_pre, n_load = 200, 600
    preload = np.full(n_pre, 0.20) + rng.normal(0, 3e-4, n_pre)
    load = 0.20 - 0.05 * np.sin(np.linspace(0, np.pi, n_load))    # never clears preload
    force = np.concatenate([preload, load])
    cycle = ["1-Preload"] * n_pre + ["1-Stretch"] * n_load

    gated = biomech.segment_qc(force, cycle, "failure", gate_snr=True)
    ungated = biomech.segment_qc(force, cycle, "pre-conditioning", gate_snr=False)

    assert gated["snr_pass"] is False
    assert gated["resid_pass"] is True and gated["range_pass"] is True
    assert gated["pass"] is False
    assert ungated["pass"] is True


# Guards 

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
    y = np.array([1.0, 2.0, 3.0])        # shorter than the minimum window
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
