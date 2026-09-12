"""Compare controllers honestly: on a frontier, not a single operating point.

Any controller can look good with the right comfort target.  So MPC is swept
across a range of ``comfort_weight`` values to trace an energy-vs-discomfort
frontier, and the question asked of it is the only fair one:

    *at the comfort the production controller actually achieved on this trip,
    how much heater energy does MPC need?*

That is read off the frontier by interpolation, which is more robust than
tuning a weight until the comfort metric happens to match.  If the frontier
does not reach the OEM's comfort level, the answer is "it cannot get there",
reported as such rather than extrapolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .controllers import MPCController, OEMReplay, PIController, Thermostat
from .cost import ComfortSpec, TripOutcome, summarise
from .env import ThermalEnv, TripScenario, make_plant, perturb_plant

__all__ = ["run_controller", "sweep_mpc", "iso_comfort_energy",
           "evaluate_trip", "robustness_sweep", "range_gain_km"]


def run_controller(env: ThermalEnv, policy, label: str) -> TripOutcome:
    if hasattr(policy, "reset"):
        policy.reset()
    t_cab, p_heat = env.rollout(policy)
    return summarise(env.scenario.trip, label, t_cab, p_heat, env.spec,
                     env.scenario.dt_s)


def sweep_mpc(env: ThermalEnv, weights, **mpc_kwargs) -> list[TripOutcome]:
    """MPC at several comfort weights -- one point on the frontier each."""
    out = []
    for w in weights:
        mpc = MPCController(comfort_weight=float(w), spec=env.spec, **mpc_kwargs)
        r = run_controller(env, mpc, f"mpc_w{w:g}")
        out.append(r)
    return out


def iso_comfort_energy(frontier: list[TripOutcome], target_discomfort: float
                       ) -> tuple[float, bool]:
    """Energy the frontier needs to match ``target_discomfort``.

    Returns ``(energy_wh, reachable)``.  ``reachable`` is False when the target
    lies outside the swept range -- an honest "cannot answer" instead of an
    extrapolation off the end of the curve.
    """
    pts = sorted(((f.discomfort_kmin, f.energy_wh) for f in frontier))
    d = np.array([p[0] for p in pts])
    e = np.array([p[1] for p in pts])
    if target_discomfort < d.min() or target_discomfort > d.max():
        return float("nan"), False
    return float(np.interp(target_discomfort, d, e)), True


def evaluate_trip(plant, scenario: TripScenario, spec: ComfortSpec | None = None,
                  weights=(0.2, 0.5, 1.0, 2.0, 5.0, 15.0), **mpc_kwargs
                  ) -> tuple[pd.DataFrame, dict]:
    """Run every controller on one trip and compute the iso-comfort saving."""
    spec = spec or ComfortSpec()
    env = ThermalEnv(plant, scenario, spec)

    baselines = [
        run_controller(env, OEMReplay(), "oem"),
        run_controller(env, Thermostat(setpoint_c=spec.setpoint_c, band_c=spec.band_c),
                       "thermostat"),
        run_controller(env, PIController(setpoint_c=spec.setpoint_c), "pi"),
    ]
    frontier = sweep_mpc(env, weights, **mpc_kwargs)

    oem = baselines[0]
    mpc_e, reachable = iso_comfort_energy(frontier, oem.discomfort_kmin)
    saving = (1.0 - mpc_e / oem.energy_wh) if (reachable and oem.energy_wh > 0) else float("nan")

    rows = [r.as_row() for r in baselines + frontier]
    summary = {
        "trip": scenario.trip,
        "oem_energy_wh": oem.energy_wh,
        "oem_discomfort_kmin": oem.discomfort_kmin,
        "mpc_energy_at_oem_comfort_wh": mpc_e,
        "reachable": reachable,
        "energy_saving_frac": saving,
    }
    return pd.DataFrame(rows), summary


def robustness_sweep(scenario: TripScenario, spec: ComfortSpec | None = None,
                     comfort_weight: float = 1.0,
                     perturbations=(("c_cab", 0.8), ("c_cab", 1.2),
                                    ("ua0", 0.8), ("ua0", 1.2),
                                    ("eta", 0.85), ("tau_h", 1.3)),
                     **mpc_kwargs) -> pd.DataFrame:
    """Tune MPC on the nominal plant, then score it on wrong ones.

    This is the check that separates a real saving from an exploited modelling
    artefact.  The controller keeps the plant it planned with; only the plant it
    is *evaluated* in is perturbed, which is exactly the sim-to-real gap.
    """
    spec = spec or ComfortSpec()
    nominal = make_plant()
    rows = []

    for name, factor in (("nominal", 1.0),) + tuple(perturbations):
        true_plant = nominal if name == "nominal" else perturb_plant(nominal, {name: factor})
        env_true = ThermalEnv(true_plant, scenario, spec)

        # the controller plans with the NOMINAL plant -- it does not know it is wrong
        mpc = MPCController(comfort_weight=comfort_weight, spec=spec, **mpc_kwargs)
        mpc_env_for_plan = ThermalEnv(nominal, scenario, spec)

        def policy(k, t_cab, _env, _mpc=mpc, _plan_env=mpc_env_for_plan):
            return _mpc(k, t_cab, _plan_env)

        mpc.reset()
        t_cab, p_heat = env_true.rollout(policy)
        r = summarise(scenario.trip, f"mpc@{name}x{factor:g}", t_cab, p_heat,
                      spec, scenario.dt_s)
        oem = run_controller(env_true, OEMReplay(), "oem")
        rows.append({**r.as_row(), "perturbation": f"{name}x{factor:g}",
                     "oem_energy_wh": oem.energy_wh,
                     "oem_discomfort_kmin": oem.discomfort_kmin})
    return pd.DataFrame(rows)


def range_gain_km(energy_saved_wh: float, distance_km: float,
                  trip_consumption_wh: float) -> float:
    """Convert a heater-energy saving into kilometres of winter range.

    Uses the trip's own measured consumption per km, so the answer is in the
    units the whole project is motivated by.  Returns NaN when the trip's
    consumption is unknown rather than assuming a fleet average.
    """
    if not (distance_km > 0 and trip_consumption_wh > 0):
        return float("nan")
    wh_per_km = trip_consumption_wh / distance_km
    return float(energy_saved_wh / wh_per_km)
