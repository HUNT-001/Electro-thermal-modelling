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

TRUE = dict(c_cab=1.2e5, ua0=62.0, ua1=3.0, ua_mass=75.0, eta=0.88,
            tau_h=25.0, cop_ac=2.2, tau_mass=1500.0, mdot_cp=48.0)
TRUE_C_MASS = TRUE["tau_mass"] * TRUE["ua_mass"]      # 112 500 J/K


def _oracle(n_trips: int = 1) -> RCPlant:
    """An RCPlant pinned to known parameters, used to generate data."""
    m = RCPlant(n_trips=n_trips)
    for prm in m.parameters():
        prm.requires_grad_(False)
    with torch.no_grad():
        m._c_cab.copy_(torch.tensor(np.log(np.expm1(TRUE["c_cab"] / 1e4))))
        m._tau_mass.copy_(torch.tensor(np.log(np.expm1(TRUE["tau_mass"] / 60.0))))
        m._mdot_cp.copy_(torch.tensor(np.log(np.expm1(TRUE["mdot_cp"]))))
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
    bad = PlantParams(c_cab=3.0, c_mass=1e5, ua0=62, ua1=3, ua_mass=75,
                      eta=0.9, tau_h=25, cop_ac=2, tau_mass=1500, mdot_cp=48)
    assert "c_cab" in bad.implausible()
    good = PlantParams(c_mass=TRUE_C_MASS, **TRUE)
    assert not good.implausible()


def test_collapsed_interior_mass_is_flagged():
    """The failure mode the winter run actually produced: C_mass ~900 J/K
    against UA_mass ~35 W/K, i.e. a 26 s 'thermal mass'."""
    collapsed = PlantParams(c_cab=70_000, c_mass=911, ua0=41, ua1=0.13, ua_mass=34.6,
                            eta=0.58, tau_h=58, cop_ac=1.5,
                            tau_mass=911 / 34.6, mdot_cp=48)
    assert "tau_mass" in collapsed.implausible()


def test_mass_time_constant_parameterisation_cannot_collapse():
    """tau_mass is the fitted quantity, so C_mass follows it and stays slow."""
    m = RCPlant(n_trips=1)
    with torch.no_grad():
        m._tau_mass.fill_(-50.0)     # push as hard as the optimiser could
        m._ua_mass.fill_(-50.0)
    p = m.params()
    assert p.tau_mass > 0 and p.c_mass > 0


def test_steady_state_power_matches_observed_holding_power():
    """~1.1-1.3 kW holds roughly 22 K in the measured urban trips."""
    p = PlantParams(c_mass=TRUE_C_MASS, **TRUE)
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
    w = build_windows(trips, horizon_s=300, stride_s=150, burn_in_s=0)
    assert len(w) > 0
    assert w.horizon == 300
    assert set(w.trips) == {"TripB01", "TripB02"}
    assert w.trip_idx.unique().numel() == 2


def test_velocity_is_converted_to_metres_per_second():
    trips = {"T": _fake_trip("T", 400)}
    w = build_windows(trips, horizon_s=300, stride_s=300, burn_in_s=0)
    kmh = trips["T"]["velocity_kmh"].to_numpy()[:300]
    assert np.allclose(w.u[0, :, 3].numpy(), kmh / 3.6, atol=1e-4)


def test_windows_with_large_gaps_are_dropped_not_imputed():
    """Filling a hole in the heater trace invents energy that never entered the cabin."""
    df = _fake_trip("T", 900)
    df.loc[100:280, "heat_power_req_w"] = np.nan      # 60% of the first window
    w = build_windows({"T": df}, horizon_s=300, stride_s=300, max_gap_frac=0.02, burn_in_s=0)
    starts = w.start_s.tolist()
    assert 0.0 not in starts, "window over the gap should have been dropped"
    assert starts == [300.0, 600.0]


def test_window_starting_on_a_missing_sample_is_dropped():
    """A rollout cannot begin from an unknown state, however small the gap."""
    df = _fake_trip("T", 900)
    df.loc[300, "heat_power_req_w"] = np.nan
    w = build_windows({"T": df}, horizon_s=300, stride_s=300, max_gap_frac=0.5, burn_in_s=0)
    assert 300.0 not in w.start_s.tolist()


def test_small_gaps_are_interpolated_and_the_window_kept():
    df = _fake_trip("T", 900)
    df.loc[150:151, "heat_power_req_w"] = np.nan      # <2% of the window
    w = build_windows({"T": df}, horizon_s=300, stride_s=300, max_gap_frac=0.02, burn_in_s=0)
    assert 0.0 in w.start_s.tolist()
    assert torch.isfinite(w.u).all()


def test_short_trips_produce_no_windows_rather_than_padding():
    with pytest.raises(ValueError, match="no windows"):
        build_windows({"T": _fake_trip("T", 100)}, horizon_s=1200, burn_in_s=0)


def test_select_trips_filters_windows():
    trips = {"A": _fake_trip("A", 800), "B": _fake_trip("B", 800, seed=2)}
    w = build_windows(trips, horizon_s=300, stride_s=200, burn_in_s=0)
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


# --------------------------------------------------------------------------
# identifiability: the duct temperature as a second observation
# --------------------------------------------------------------------------

def test_vent_temperature_reflects_delivered_heat():
    """T_vent - T_cab must be proportional to Q_del, which is what makes it
    an observation of the heater path rather than of the cabin."""
    m = _oracle()
    state = torch.tensor([[20.0, 20.0, 0.0], [20.0, 20.0, 3500.0]])
    vent = m.vent_temp(state)
    assert float(vent[0]) == pytest.approx(20.0)
    assert float(vent[1] - vent[0]) == pytest.approx(3500.0 / TRUE["mdot_cp"], rel=1e-4)


def test_vent_term_enters_the_loss_only_when_observed():
    from etm.plant.fit import rollout_loss
    w, oracle = _synthetic_windows(n_windows=4, steps=200, seed=21)
    assert not w.has_vent
    bare = float(rollout_loss(oracle, w, vent_weight=0.5))

    with torch.no_grad():
        state0 = oracle.initial_state(w.y[:, 0], w.u[:, 0, 0])
        traj = oracle.rollout(state0, w.u, w.trip_idx)
        w.y_vent = oracle.vent_temp(traj) + 5.0      # deliberately wrong by 5 K
    assert w.has_vent
    assert float(rollout_loss(oracle, w, vent_weight=0.5)) > bare


def test_plausibility_penalty_is_zero_inside_the_ranges_and_positive_outside():
    from etm.plant.fit import plausibility_penalty
    assert float(plausibility_penalty(_oracle())) == pytest.approx(0.0, abs=1e-6)
    bad = RCPlant(n_trips=1)
    with torch.no_grad():
        bad._tau_mass.fill_(-8.0)            # drive the mass time constant to ~0
    assert float(plausibility_penalty(bad)) > 0.1


def test_windows_pick_up_the_duct_temperature_when_present():
    trips = {"T": _fake_trip("T", 800)}
    trips["T"]["hx_out_c"] = trips["T"]["cabin_temp_c"] + 30.0
    w = build_windows(trips, horizon_s=300, stride_s=300, burn_in_s=0)
    assert w.has_vent
    assert w.y_vent.shape == w.y.shape
    assert torch.allclose(w.y_vent - w.y, torch.full_like(w.y, 30.0), atol=1e-3)


def test_windows_have_no_duct_temperature_on_the_reduced_summer_schema():
    """TripA files carry no heat-exchanger or vent channels at all."""
    w = build_windows({"T": _fake_trip("T", 800)}, horizon_s=300, stride_s=300, burn_in_s=0)
    assert not w.has_vent
    assert w.truncate(100).y_vent is None
    assert w.subset(torch.tensor([True, True])).y_vent is None


# --------------------------------------------------------------------------
# burn-in: the latent states are guessed, so don't score the guess
# --------------------------------------------------------------------------

def test_burn_in_is_simulated_but_not_scored():
    trips = {"T": _fake_trip("T", 1200)}
    w = build_windows(trips, horizon_s=600, stride_s=600, burn_in_s=200)
    assert w.burn_in == 200
    assert w.horizon == 600                 # scored length
    assert w.u.shape[1] == 800              # simulated length
    assert w.y[:, w.scored].shape[1] == 600


def test_truncate_keeps_the_burn_in_prefix():
    """Shortening to a 60 s horizon must not throw away the settling period."""
    trips = {"T": _fake_trip("T", 1200)}
    w = build_windows(trips, horizon_s=600, stride_s=600, burn_in_s=200)
    t = w.truncate(60)
    assert t.burn_in == 200
    assert t.horizon == 60
    assert t.u.shape[1] == 260


def test_observer_recovers_the_fast_state_but_not_the_slow_one():
    """The result that sent L1 to trip-start anchoring.

    Running an observer over measured history nails Q_del, whose 45 s time
    constant lets it forget its initial value, and cannot recover T_mass, whose
    25 min time constant means five minutes of history carries almost no
    information about it. No initialisation trick fixes an unobservable state;
    only starting where the state is known does.
    """
    from etm.plant.windows import WindowSet
    oracle = _oracle()
    u = _excited_inputs(6, 1200, seed=31)
    idx = torch.zeros(6, dtype=torch.long)
    with torch.no_grad():
        state0 = torch.stack([torch.full((6,), 5.0), torch.full((6,), 20.0),
                              torch.zeros(6)], dim=-1)
        traj = oracle.rollout(state0, u, idx)
        y = traj[..., 0]
        est = oracle.initial_state_from_history(y[:, :300], u[:, :300])

    assert float(est[0, 2]) == pytest.approx(float(traj[0, 299, 2]), rel=0.01)
    assert abs(float(est[0, 1]) - float(traj[0, 299, 1])) > 5.0

    # and consequently a mid-trip window scores worse than one anchored where
    # the interior state is genuinely known
    mid = WindowSet(u=u, y=y, trip_idx=idx, start_s=torch.zeros(6),
                    trips=["t"], burn_in=300)
    assert rollout_rmse(oracle, mid, 300) > 0.0


def test_initial_state_from_history_recovers_a_lagging_interior():
    """T_mass is estimated as an EWMA of measured cabin temperature over
    roughly tau_mass, because the interior soaks toward where the air has been."""
    m = _oracle()
    b = 900
    y_hist = torch.full((1, b), 10.0)
    y_hist[:, b // 2:] = 25.0              # cabin stepped up halfway through
    u_hist = torch.zeros(1, b, 4)
    state = m.initial_state_from_history(y_hist, u_hist)
    t_cab0, t_mass0, q_del0 = state[0]
    assert float(t_cab0) == pytest.approx(25.0)      # cabin is measured
    assert 10.0 < float(t_mass0) < 25.0              # interior lags behind it
    assert float(q_del0) == pytest.approx(0.0, abs=1e-3)


def test_initial_state_from_history_settles_the_heater_lag():
    """Q_del has a short time constant, so running it through the history
    lands it on the steady value regardless of where it started."""
    m = _oracle()
    b = 600
    u_hist = torch.zeros(1, b, 4)
    u_hist[..., 0] = 6000.0
    state = m.initial_state_from_history(torch.full((1, b), 20.0), u_hist)
    assert float(state[0, 2]) == pytest.approx(TRUE["eta"] * 6000.0, rel=0.02)


def test_unidentifiable_freezes_cop_when_the_compressor_never_runs():
    """A/C is active in 0-6 % of winter rows; a COP fitted from that is noise."""
    from etm.plant.fit import unidentifiable
    w, _ = _synthetic_windows(n_windows=4, steps=300, seed=41)   # p_ac is all zeros
    assert "cop_ac" in unidentifiable(w)

    w.u[..., 1] = 1200.0
    assert "cop_ac" not in unidentifiable(w)


def test_unidentifiable_freezes_speed_term_without_speed_variation():
    from etm.plant.fit import unidentifiable
    w, _ = _synthetic_windows(n_windows=4, steps=300, seed=42)
    w.u[..., 3] = 10.0                       # constant speed: UA1 unconstrained
    assert "ua1" in unidentifiable(w)


def test_frozen_parameters_do_not_move_during_fitting():
    w, _ = _synthetic_windows(n_windows=4, steps=200, seed=43)
    cfg = FitConfig(horizons_s=(120,), epochs_per_stage=15, batch_size=4, progress=False)
    res = fit_plant(w, cfg=cfg)
    assert "cop_ac" in res.frozen
    assert res.model.params().cop_ac == pytest.approx(RCPlant.INITIAL["cop_ac"], rel=1e-4)


def test_trip_start_anchor_gives_one_window_per_trip_from_key_on():
    """At key-on the interior and the cabin air are both at ambient -- the one
    moment the unobservable slow state is actually known."""
    trips = {"A": _fake_trip("A", 3000), "B": _fake_trip("B", 3000, seed=5)}
    w = build_windows(trips, horizon_s=1200, anchor="trip_start", burn_in_s=0)
    assert len(w) == 2
    assert w.start_s.tolist() == [0.0, 0.0]
    assert sorted(w.trip_idx.tolist()) == [0, 1]


def test_trip_start_anchor_skips_trips_shorter_than_the_horizon():
    trips = {"A": _fake_trip("A", 3000), "SHORT": _fake_trip("SHORT", 400, seed=6)}
    w = build_windows(trips, horizon_s=1200, anchor="trip_start", burn_in_s=0)
    assert len(w) == 1
    assert w.trips[w.trip_idx[0]] == "A"


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError, match="anchor must be"):
        build_windows({"A": _fake_trip("A", 2000)}, horizon_s=600, anchor="middle")


def test_interior_state_is_unobservable_from_a_short_history():
    """Why trip_start exists: with tau_mass ~25 min, five minutes of measured
    cabin temperature cannot recover the interior state, however it is
    estimated. The observer nails the fast state and misses the slow one."""
    m = _oracle()
    u = _excited_inputs(1, 900, seed=71)
    idx = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        truth0 = torch.tensor([[5.0, 20.0, 0.0]])       # interior 15 K above the air
        traj = m.rollout(truth0, u, idx)
        est = m.initial_state_from_history(traj[:, :300, 0], u[:, :300])
    true_at_300 = traj[0, 299]
    assert float(est[0, 2]) == pytest.approx(float(true_at_300[2]), rel=0.01)   # Q_del: fine
    assert abs(float(est[0, 1]) - float(true_at_300[1])) > 5.0                  # T_mass: not


def test_ua1_frozen_when_no_window_sustains_highway_speed():
    """The winter failure: global speed varies but no window pairs high speed
    with a warm cabin, so UA1 has no gradient."""
    from etm.plant.fit import unidentifiable
    w, _ = _synthetic_windows(n_windows=4, steps=300, seed=61)
    w.u[..., 3] = 8.0                      # urban crawl throughout
    assert "ua1" in unidentifiable(w)


def test_ua1_identifiable_when_a_window_holds_highway_speed():
    from etm.plant.fit import unidentifiable
    w, _ = _synthetic_windows(n_windows=4, steps=300, seed=62)
    w.u[..., 3] = 8.0
    w.u[0, :, 3] = 28.0                    # one window is a motorway leg
    assert "ua1" not in unidentifiable(w)


# --------------------------------------------------------------------------
# steady-state UA identification (the UA1 finding)
# --------------------------------------------------------------------------

def _steady_trip(name, n=3000, ua0=40.0, ua1=0.0, amb=3.0, aux=300.0,
                 eta=0.96, speed_ms=15.0, seed=0):
    """A synthetic warm-cabin holding trip with a KNOWN envelope conductance."""
    rng = np.random.default_rng(seed)
    cab = 22.0 + rng.normal(0, 0.02, n).cumsum() * 0.001    # near setpoint, nearly flat
    cab = np.clip(cab, 21.0, 23.0)
    v = np.abs(rng.normal(speed_ms, 5.0, n))
    dT = cab - amb
    # invert the steady balance to get the power that would hold this cabin
    p = ((ua0 + ua1 * v) * dT - aux) / eta
    p = np.clip(p + rng.normal(0, 20, n), 200, 7000)
    return pd.DataFrame({
        "trip": name, "time_s": np.arange(n, dtype=float),
        "cabin_temp_c": cab, "amb_temp_c": np.full(n, amb),
        "velocity_kmh": v * 3.6, "heat_power_req_w": p,
    })


def test_steady_state_recovers_a_known_ua1():
    """If the data really has speed-dependent loss, the regression must find it."""
    from etm.plant.steady_state import identify_ua_steady_state
    trips = {f"T{i}": _steady_trip(f"T{i}", ua0=40.0, ua1=3.0,
                                   speed_ms=10.0 + 6 * i, seed=i) for i in range(6)}
    res = identify_ua_steady_state(trips)
    assert res.ua1_within_trip == pytest.approx(3.0, abs=0.8), res.summary()
    assert res.speed_dependence_detected


def test_steady_state_reports_zero_when_loss_is_speed_independent():
    """The real-data case: no speed dependence must read as zero, not as noise
    dressed up as a small positive number."""
    from etm.plant.steady_state import identify_ua_steady_state
    trips = {f"T{i}": _steady_trip(f"T{i}", ua0=40.0, ua1=0.0,
                                   speed_ms=8.0 + 5 * i, seed=i) for i in range(6)}
    res = identify_ua_steady_state(trips)
    assert abs(res.ua1_within_trip) < 0.5, res.summary()
    assert not res.speed_dependence_detected


def test_steady_state_mask_selects_warm_flat_heater_on_samples():
    from etm.plant.steady_state import steady_state_mask
    n = 600
    df = pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "cabin_temp_c": np.concatenate([np.linspace(2, 22, 300), np.full(300, 22.0)]),
        "amb_temp_c": np.full(n, 3.0),
        "velocity_kmh": np.full(n, 40.0),
        "heat_power_req_w": np.full(n, 1500.0),
    })
    m = steady_state_mask(df)
    assert m[:250].sum() == 0            # warm-up phase excluded (still rising)
    assert m[350:].sum() > 200           # the holding phase is selected


def test_steady_state_needs_a_real_temperature_gradient():
    """Dividing by a near-zero cabin-ambient gap is where this analysis breaks;
    the mask must refuse those samples."""
    from etm.plant.steady_state import steady_state_mask
    n = 400
    df = pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "cabin_temp_c": np.full(n, 22.0),
        "amb_temp_c": np.full(n, 21.0),     # only 1 K gap -- below min_dt_c
        "velocity_kmh": np.full(n, 30.0),
        "heat_power_req_w": np.full(n, 1500.0),
    })
    assert steady_state_mask(df).sum() == 0
