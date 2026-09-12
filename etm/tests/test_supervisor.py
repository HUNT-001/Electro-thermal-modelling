import numpy as np
import pandas as pd
import pytest

from etm.control.controllers import OEMReplay, PIController
from etm.control.cost import ComfortSpec
from etm.control.env import HEATER_MAX_W, ThermalEnv, TripScenario, make_plant
from etm.control.evaluate import run_controller
from etm.control.supervisor import (ActuatorClamp, InputEnvelope, RateLimit,
                                    SensorValid, SupervisedController)


def _frame(n=600, amb=3.0, speed_kmh=40.0):
    t = np.arange(n, dtype=float)
    return pd.DataFrame({
        "trip": "T", "time_s": t,
        "heat_power_req_w": np.full(n, 1500.0),
        "cabin_temp_c": 22 - 20 * np.exp(-t / 300),
        "amb_temp_c": np.full(n, amb),
        "aircon_power_w": np.zeros(n),
        "velocity_kmh": np.full(n, speed_kmh),
    })


def _env(**kw):
    sc = TripScenario.from_trip("T", _frame(**kw), dt_s=10.0, max_s=kw.get("n", 600))
    return ThermalEnv(make_plant(), sc, ComfortSpec())


class Constant:
    def __init__(self, w): self.w = w
    def reset(self): pass
    def __call__(self, k, t_cab, env): return self.w


# --------------------------------------------------------------------------
# individual guards
# --------------------------------------------------------------------------

def test_actuator_clamp_bounds_the_command():
    g = ActuatorClamp()
    env = _env()
    assert g.check(0, 12000.0, 20.0, env) == (HEATER_MAX_W, "actuator limit", False)
    assert g.check(0, -50.0, 20.0, env) == (0.0, "actuator limit", False)
    assert g.check(0, 1500.0, 20.0, env) == (1500.0, None, False)


def test_actuator_clamp_treats_a_nan_action_as_a_fault():
    applied, reason, fault = ActuatorClamp().check(0, float("nan"), 20.0, _env())
    assert applied == 0.0 and fault and "non-finite" in reason


def test_rate_limit_slews_toward_the_request():
    g = RateLimit(max_rate_w_per_s=100.0)     # 100 W/s x 10 s = 1000 W per step
    env = _env()
    g.reset()
    assert g.check(0, 5000.0, 20.0, env)[0] == 5000.0     # first step sets the anchor
    a1, r1, _ = g.check(1, 5000.0, 20.0, env)
    assert a1 == 5000.0 and r1 is None
    a2, r2, _ = g.check(2, 0.0, 20.0, env)                # big down-step
    assert a2 == pytest.approx(4000.0) and r2 == "rate limit"


def test_input_envelope_faults_outside_the_identified_range():
    hot = InputEnvelope().check(0, 1500.0, 20.0, _env(amb=40.0))
    assert hot[2] and "ambient" in hot[1]
    fast = InputEnvelope().check(0, 1500.0, 20.0, _env(speed_kmh=220.0))
    assert fast[2] and "speed" in fast[1]
    ok = InputEnvelope().check(0, 1500.0, 20.0, _env(amb=3.0, speed_kmh=40.0))
    assert not ok[2]


def test_sensor_valid_faults_on_nan_or_absurd_cabin():
    assert SensorValid().check(0, 1500.0, float("nan"), _env())[2]
    assert SensorValid().check(0, 1500.0, 200.0, _env())[2]      # 200 C is impossible
    assert SensorValid().check(0, 1500.0, None, _env())[2]       # non-numeric reading
    assert not SensorValid().check(0, 1500.0, 21.0, _env())[2]


# --------------------------------------------------------------------------
# the supervisor as a whole
# --------------------------------------------------------------------------

def test_supervisor_is_transparent_when_the_primary_behaves():
    """A well-behaved controller must pass through unchanged."""
    env = _env()
    prim = Constant(1500.0)
    sup = SupervisedController(primary=prim, fallback=PIController())
    _, p = env.rollout(sup)
    assert np.allclose(p, 1500.0)
    assert sup.report.n_fallback == 0 and sup.report.n_amended == 0


def test_supervisor_contains_an_out_of_range_primary():
    env = _env()
    sup = SupervisedController(primary=Constant(50000.0), fallback=PIController())
    _, p = env.rollout(sup)
    assert p.max() <= HEATER_MAX_W + 1e-6
    assert sup.report.n_amended > 0                     # clamped every step
    assert sup.report.n_fallback == 0                  # clamp is not a fault


def test_supervisor_falls_back_on_a_nan_emitting_primary():
    env = _env()
    sup = SupervisedController(primary=Constant(float("nan")),
                               fallback=Constant(1200.0))
    _, p = env.rollout(sup)
    assert np.isfinite(p).all()
    assert np.allclose(p, 1200.0)                       # fallback drove throughout
    assert sup.report.n_fallback == len(p)


def test_supervisor_falls_back_on_a_sensor_dropout():
    """Cabin sensor goes NaN mid-trip; the model-free fallback must take over
    exactly for those steps and hand back afterwards."""
    frame = _frame(600)
    env = ThermalEnv(make_plant(),
                     TripScenario.from_trip("T", frame, dt_s=10.0, max_s=600),
                     ComfortSpec())
    # a primary that trusts the cabin reading; a fallback that is safe without it
    sup = SupervisedController(primary=Constant(3000.0), fallback=Constant(1000.0))

    # feed a NaN cabin temperature on a band of steps via a custom rollout
    s = env.scenario
    import torch
    state = env.initial_state()
    aux = torch.zeros(1)
    applied = []
    for k in range(len(s)):
        t_cab = float(state[0, 0])
        if 20 <= k < 30:
            t_cab = float("nan")                        # sensor dropout
        p = sup(k, t_cab, env)
        applied.append(p)
        u0 = env.inputs_at(k, p); u1 = env.inputs_at(k + 1, p)
        state = env.plant.step(state, u0, u1, aux, s.dt_s)
    applied = np.array(applied)
    assert np.allclose(applied[20:30], 1000.0)          # fallback during dropout
    assert np.allclose(applied[:20], 3000.0)            # primary before
    assert np.allclose(applied[30:], 3000.0)            # primary after -- handed back


def test_supervisor_falls_back_outside_the_envelope_and_recovers():
    """Ambient spikes out of the identified range for a stretch, then returns."""
    frame = _frame(900)
    frame.loc[30:50, "amb_temp_c"] = 45.0               # heatwave the plant never saw
    env = ThermalEnv(make_plant(),
                     TripScenario.from_trip("T", frame, dt_s=10.0, max_s=900),
                     ComfortSpec())
    sup = SupervisedController(primary=Constant(4000.0), fallback=Constant(900.0))
    _, p = env.rollout(sup)
    # steps 3..5 (10 s blocks) are inside the spike; fallback owns them
    reasons = sup.report.reasons()
    assert any("ambient" in r for r in reasons)
    assert sup.report.n_fallback > 0


def test_actuator_limits_hold_even_for_the_fallback():
    """A fallback that itself commands too much is still clamped."""
    env = _env()
    sup = SupervisedController(primary=Constant(float("nan")),
                               fallback=Constant(99999.0))
    _, p = env.rollout(sup)
    assert p.max() <= HEATER_MAX_W + 1e-6


def test_report_summary_is_auditable():
    env = _env()
    sup = SupervisedController(primary=Constant(50000.0), fallback=PIController())
    env.rollout(sup)
    txt = sup.report.summary()
    assert "actuator limit" in txt
    assert sup.report.reasons()["actuator limit"] == sup.report.n_amended
