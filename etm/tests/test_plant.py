import numpy as np
import pandas as pd
import pytest
import torch

from etm.plant.evaluate import (energy_balance_check, evaluate_plant,
                                exponential_decay_rmse, persistence_rmse)
from etm.plant.fit import FitConfig, fit_plant, rollout_rmse, transfer_aux
from etm.plant.model import PLAUSIBLE_RANGES, PlantParams, RCPlant
from etm.plant.windows import build_windows


# --------------------------------------------------------------------------
# synthetic ground truth
# --------------------------------------------------------------------------

TRUE = dict(c_cab=1.2e5, c_mass=5.0e5, ua0=62.0, ua1=3.0,
            ua_mass=75.0, eta=0.88, tau_h=25.0, cop_ac=2.2)


def _oracle(n_trips: int = 1) -> RCPlant:
    """An RCPlant pinned to known parameters, used to generate data."""
    m = RCPlant(n_trips=n_trips)
    for prm in m.parameters():
        prm.requires_grad_(False)
    with torch.no_grad():
        m._c_cab.copy_(torch.tensor(np.log(np.expm1(TRUE["c_cab"] / 1e4))))
        m._c_mass.copy_(torch.tensor(np.log(np.expm1(TRUE["c_mass"] / 1e4))))
        m._ua0.copy_(torch.tensor(np.log(np.expm1(TRUE["ua0"]))))
        m._ua1.copy_(torch.tensor(np.log(np.expm1(TRUE["ua1"]))))
        m._ua_mass.copy_(torch.tensor(np.log(np.expm1(TRUE["ua_mass"]))))
        m._tau_h.copy_(torch.tensor(np.log(np.expm1(TRUE["tau_h"] - 1.0))))
        m._cop_ac.copy_(torch.tensor(np.log(np.expm1(TRUE["cop_ac"]))))
        lo, hi = PLAUSIBLE_RANGES["eta"]
        p = (TRUE["eta"] - lo) / (hi - lo)
        m._eta.copy_(torch.tensor(np.log(p / (1 - p))))
        m._aux.zero_()
    return m


def _excited_inputs(n_windows: int, steps: int, seed: int = 0) -> torch.Tensor:
    """Heater / ambient / speed traces with enough excitation to identify from.

    Real winter trips look like this: a full-power warm-up, a decay to a
    holding power, and speed varying with traffic.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    u = np.zeros((n_windows, steps, 4))
    for i in range(n_windows):
        warm = 7000 * np.exp(-t / rng.uniform(200, 500)) + rng.uniform(800, 1500)
        square = 1200 * (np.sin(2 * np.pi * t / rng.uniform(150, 400)) > 0)
        u[i, :, 0] = np.clip(warm + square, 0, 7000)
        u[i, :, 1] = 0.0
        u[i, :, 2] = rng.uniform(-3, 9)
        speed = np.abs(rng.normal(12, 6, size=steps))
        u[i, :, 3] = np.convolve(speed, np.ones(30) / 30, mode="same")
    return torch.tensor(u, dtype=torch.float32)


def _synthetic_windows(n_windows=24, steps=900, seed=0, noise_c=0.0):
    from etm.plant.windows import WindowSet
    oracle = _oracle()
    u = _excited_inputs(n_windows, steps, seed)
    trip_idx = torch.zeros(n_windows, dtype=torch.long)
    t0 = torch.tensor(np.random.default_rng(seed).uniform(-3, 9, n_windows), dtype=torch.float32)
    with torch.no_grad():
        state0 = oracle.initial_state(t0, u[:, 0, 0])
        y = oracle.rollout(state0, u, trip_idx)[..., 0]
    if noise_c:
        y = y + noise_c * torch.randn_like(y)
    return WindowSet(u=u, y=y, trip_idx=trip_idx,
                     start_s=torch.zeros(n_windows), trips=["synthetic"]), oracle


# --------------------------------------------------------------------------
# integrator
# --------------------------------------------------------------------------

def test_no_input_no_gradient_means_no_change():
    """With cabin, mass and ambient all equal and the heater off, nothing moves."""
    m = _oracle()
    state = torch.tensor([[20.0, 20.0, 0.0]])
    u = torch.tensor([[0.0, 0.0, 20.0, 0.0]])
    d = m.derivatives(state, u, torch.zeros(1))
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def test_cabin_warms_when_heat_is_delivered():
    m = _oracle()
    state = torch.tensor([[0.0, 0.0, 3000.0]])
    u = torch.tensor([[5000.0, 0.0, 0.0, 0.0]])
    assert float(m.derivatives(state, u, torch.zeros(1))[0, 0]) > 0


def test_cabin_cools_when_aircon_runs():
    m = _oracle()
    state = torch.tensor([[25.0, 25.0, 0.0]])
    u = torch.tensor([[0.0, 1500.0, 25.0, 0.0]])
    assert float(m.derivatives(state, u, torch.zeros(1))[0, 0]) < 0


def test_speed_increases_envelope_loss():
    """UA(v) must grow with road speed, or the model cannot explain highway trips."""
    m = _oracle()
    state = torch.tensor([[22.0, 22.0, 0.0], [22.0, 22.0, 0.0]])
    u = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 30.0]])
    d = m.derivatives(state, u, torch.zeros(2))[:, 0]
    assert float(d[1]) < float(d[0])


def test_heater_lag_is_first_order_with_the_right_time_constant():
    """Q_del must reach 1 - 1/e of its steady value after tau_h seconds."""
    m = _oracle()
    steps = int(TRUE["tau_h"]) + 1
    u = torch.zeros(1, steps, 4)
    u[..., 0] = 7000.0
    state0 = torch.tensor([[0.0, 0.0, 0.0]])
    traj = m.rollout(state0, u, torch.zeros(1, dtype=torch.long))
    final = TRUE["eta"] * 7000.0
    assert float(traj[0, -1, 2]) == pytest.approx(final * (1 - np.exp(-1)), rel=0.05)


def test_rk4_matches_analytic_first_order_response():
    """Integrator accuracy, checked against a closed-form solution."""
    m = _oracle()
    steps = 200
    u = torch.zeros(1, steps, 4)
    u[..., 0] = 7000.0
    traj = m.rollout(torch.tensor([[0.0, 0.0, 0.0]]), u, torch.zeros(1, dtype=torch.long))
    t = np.arange(steps)
    analytic = TRUE["eta"] * 7000.0 * (1 - np.exp(-t / TRUE["tau_h"]))
    assert np.allclose(traj[0, :, 2].numpy(), analytic, atol=1.0)


def test_energy_is_conserved_over_a_rollout():
    w, oracle = _synthetic_windows(n_windows=4, steps=600)
    report = energy_balance_check(oracle, w)
    assert report["max_rel_error"] < 1e-2, report


# --------------------------------------------------------------------------
# identification -- the test that matters
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_identification_recovers_known_parameters():
    """Generate from known physics, fit, and check the parameters come back.

    This is the only test that can distinguish "the optimiser reduced a loss"
    from "the model identified the system". Without it a plant that has folded
    the heater lag into the cabin capacitance still looks like a success.
    """
    w, _ = _synthetic_windows(n_windows=24, steps=900, seed=1)
    cfg = FitConfig(horizons_s=(120, 300, 900), epochs_per_stage=120, lr=5e-2,
                    batch_size=24, log_every=10_000)
    got = fit_plant(w, cfg=cfg).model.params()

    # Steady-state gain and the fast time constant are what a controller needs
    # to be right; they are also the best-conditioned directions in the fit.
    assert got.ua0 == pytest.approx(TRUE["ua0"], rel=0.30), got
    assert got.c_cab == pytest.approx(TRUE["c_cab"], rel=0.35), got
    assert got.tau_h == pytest.approx(TRUE["tau_h"], rel=0.60), got
    assert not got.implausible(), got.implausible()


@pytest.mark.slow
def test_fit_beats_both_baselines_on_synthetic_data():
    w, _ = _synthetic_windows(n_windows=20, steps=900, seed=2, noise_c=0.05)
    cfg = FitConfig(horizons_s=(120, 300, 900), epochs_per_stage=90, lr=5e-2,
                    log_every=10_000)
    scores = evaluate_plant(fit_plant(w, cfg=cfg).model, w, horizons=(60, 300))
    for h in (60, 300):
        assert scores.skill(h) > 0.5, scores.table()


# --------------------------------------------------------------------------
# parameters and plausibility
# --------------------------------------------------------------------------

def test_parameters_stay_positive_under_adversarial_gradients():
    """Softplus parameterisation must make a negative capacitance unreachable."""
    m = RCPlant(n_trips=1)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(-50.0)
    p = m.params()
    assert p.c_cab > 0 and p.c_mass > 0 and p.ua0 > 0 and p.tau_h > 0


def test_implausible_flags_a_nonsense_fit():
    bad = PlantParams(c_cab=3.0, c_mass=1e5, ua0=62, ua1=3,
                      ua_mass=75, eta=0.9, tau_h=25, cop_ac=2)
    assert "c_cab" in bad.implausible()
    good = PlantParams(**TRUE)
    assert not good.implausible()


def test_steady_state_power_matches_observed_holding_power():
    """~1.1-1.3 kW holds roughly 22 K in the measured urban trips."""
    p = PlantParams(**TRUE)
    assert 1000 < p.steady_state_power_w(delta_t=22.0, speed_ms=0.0) < 2000


def test_aux_term_is_bounded():
    """The per-trip auxiliary heat must not be able to act as a second heater."""
    m = RCPlant(n_trips=3, aux_max_w=400.0)
    with torch.no_grad():
        m._aux.fill_(1e6)
    assert float(m.aux_w(torch.tensor([0])).abs().max()) <= 400.0 + 1e-3


def test_transfer_aux_keeps_physics_and_zeroes_trip_terms():
    """A held-out trip has no fitted auxiliary term -- that is the honest setting."""
    m = RCPlant(n_trips=5)
    with torch.no_grad():
        m._aux.fill_(2.0)
    fresh = transfer_aux(m, n_trips=2)
    assert fresh.params() == m.params()
    assert float(fresh.aux_w(torch.tensor([0, 1])).abs().max()) == 0.0


# --------------------------------------------------------------------------
# baselines and windowing
# --------------------------------------------------------------------------

def test_persistence_is_strong_at_short_horizons():
    """The reason one-step metrics are meaningless: doing nothing scores well."""
    w, _ = _synthetic_windows(n_windows=8, steps=600, seed=3)
    assert persistence_rmse(w, 60) < persistence_rmse(w, 600)


def test_exponential_baseline_reduces_to_persistence_for_a_long_time_constant():
    """Sanity: with tau -> infinity the decay baseline *is* persistence."""
    w, _ = _synthetic_windows(n_windows=8, steps=600, seed=4)
    assert exponential_decay_rmse(w, 600, tau_s=1e9) == pytest.approx(
        persistence_rmse(w, 600), rel=1e-3)


def test_exponential_baseline_wins_on_cooling_but_not_on_warm_up():
    """Which baseline is harder depends on the regime, so evaluation keeps both.

    Decaying toward ambient is the strong baseline when the cabin is cooling
    down.  During a heater warm-up the cabin moves *away* from ambient, and
    plain persistence wins instead.  Reporting only one baseline would flatter
    the plant model in whichever regime that baseline happens to be weak.
    """
    from etm.plant.windows import WindowSet
    oracle = _oracle()
    steps = 900
    u = torch.zeros(1, steps, 4)          # heater off, cold outside
    u[..., 2] = -5.0
    with torch.no_grad():
        state0 = torch.tensor([[22.0, 22.0, 0.0]])
        y = oracle.rollout(state0, u, torch.zeros(1, dtype=torch.long))[..., 0]
    cooling = WindowSet(u=u, y=y, trip_idx=torch.zeros(1, dtype=torch.long),
                        start_s=torch.zeros(1), trips=["cooling"])
    assert exponential_decay_rmse(cooling, 900) < persistence_rmse(cooling, 900)

    warmup, _ = _synthetic_windows(n_windows=8, steps=900, seed=4)
    assert persistence_rmse(warmup, 900) < exponential_decay_rmse(warmup, 900)


def _fake_trip(name, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "trip": name, "time_s": np.arange(n, dtype=float),
        "cabin_temp_c": 20 + rng.normal(0, 0.1, n).cumsum() * 0.01,
        "heat_power_req_w": rng.uniform(0, 7000, n),
        "aircon_power_w": np.zeros(n),
        "amb_temp_c": np.full(n, 4.0),
        "velocity_kmh": rng.uniform(0, 60, n),
    })


def test_windows_never_straddle_two_trips():
    trips = {"TripB01": _fake_trip("TripB01", 1000),
             "TripB02": _fake_trip("TripB02", 1000, seed=1)}
    w = build_windows(trips, horizon_s=300, stride_s=150)
    assert len(w) > 0
    assert w.horizon == 300
    assert set(w.trips) == {"TripB01", "TripB02"}
    assert w.trip_idx.unique().numel() == 2


def test_velocity_is_converted_to_metres_per_second():
    trips = {"T": _fake_trip("T", 400)}
    w = build_windows(trips, horizon_s=300, stride_s=300)
    kmh = trips["T"]["velocity_kmh"].to_numpy()[:300]
    assert np.allclose(w.u[0, :, 3].numpy(), kmh / 3.6, atol=1e-4)


def test_windows_with_large_gaps_are_dropped_not_imputed():
    """Filling a hole in the heater trace invents energy that never entered the cabin."""
    df = _fake_trip("T", 900)
    df.loc[100:280, "heat_power_req_w"] = np.nan      # 60% of the first window
    w = build_windows({"T": df}, horizon_s=300, stride_s=300, max_gap_frac=0.02)
    starts = w.start_s.tolist()
    assert 0.0 not in starts, "window over the gap should have been dropped"
    assert starts == [300.0, 600.0]


def test_window_starting_on_a_missing_sample_is_dropped():
    """A rollout cannot begin from an unknown state, however small the gap."""
    df = _fake_trip("T", 900)
    df.loc[300, "heat_power_req_w"] = np.nan
    w = build_windows({"T": df}, horizon_s=300, stride_s=300, max_gap_frac=0.5)
    assert 300.0 not in w.start_s.tolist()


def test_small_gaps_are_interpolated_and_the_window_kept():
    df = _fake_trip("T", 900)
    df.loc[150:151, "heat_power_req_w"] = np.nan      # <2% of the window
    w = build_windows({"T": df}, horizon_s=300, stride_s=300, max_gap_frac=0.02)
    assert 0.0 in w.start_s.tolist()
    assert torch.isfinite(w.u).all()


def test_short_trips_produce_no_windows_rather_than_padding():
    with pytest.raises(ValueError, match="no windows"):
        build_windows({"T": _fake_trip("T", 100)}, horizon_s=1200)


def test_select_trips_filters_windows():
    trips = {"A": _fake_trip("A", 800), "B": _fake_trip("B", 800, seed=2)}
    w = build_windows(trips, horizon_s=300, stride_s=200)
    only_a = w.select_trips({"A"})
    assert len(only_a) < len(w)
    assert set(only_a.trip_idx.tolist()) == {w.trips.index("A")}


def test_evaluation_always_scores_the_full_window_horizon():
    """With 900s windows, a fixed (60, 300, 1200) list would report only the
    two horizons where doing nothing already works."""
    w, oracle = _synthetic_windows(n_windows=4, steps=900, seed=7)
    scores = evaluate_plant(oracle, w, horizons=(60, 300, 1200))
    assert 900 in scores.plant
    assert 1200 not in scores.plant
    assert set(scores.plant) == set(scores.persistence) == set(scores.exponential)


def test_movement_from_initial_detects_an_unfitted_model():
    """Cross-fold agreement is not identification if nothing ever moved.

    A fresh model agrees with itself perfectly across every fold and has
    learned nothing; movement_from_initial is what tells the two apart.
    """
    fresh = RCPlant(n_trips=1)
    assert max(fresh.movement_from_initial().values()) < 1e-5

    moved = _oracle()          # different parameters entirely
    m = moved.movement_from_initial()
    assert m["ua0"] > 0.1 and m["c_cab"] > 0.1


def test_epochs_can_taper_per_curriculum_stage():
    """Cost is linear in horizon x epochs, so the long stage must be cheapenable."""
    cfg = FitConfig(horizons_s=(120, 300, 1200), epochs_per_stage=(400, 200, 50))
    assert cfg.epochs_for(120) == 400
    assert cfg.epochs_for(1200) == 50
    flat = FitConfig(horizons_s=(120, 300), epochs_per_stage=99)
    assert flat.epochs_for(120) == flat.epochs_for(300) == 99


def test_runtime_estimate_scales_with_epochs():
    from etm.plant.fit import estimate_runtime
    w, _ = _synthetic_windows(n_windows=4, steps=300, seed=11)
    cheap = FitConfig(horizons_s=(120,), epochs_per_stage=10, batch_size=4, progress=False)
    dear = FitConfig(horizons_s=(120,), epochs_per_stage=100, batch_size=4, progress=False)
    assert estimate_runtime(w, dear) > 3 * estimate_runtime(w, cheap)
    assert estimate_runtime(w, cheap, n_folds=5) > estimate_runtime(w, cheap)
