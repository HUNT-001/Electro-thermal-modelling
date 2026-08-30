"""Score the identified plant against baselines it has to beat.

A rollout RMSE quoted on its own is not a result.  Cabin temperature moves
slowly, so "the temperature stays where it is" is already a strong predictor
over a minute, and "it decays exponentially toward ambient" is strong over ten.
The plant model earns its place only by beating both -- and by beating them at
long horizons, since that is the regime a controller plans in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .fit import rollout_rmse
from .model import RCPlant
from .windows import WindowSet

__all__ = ["BaselineScores", "evaluate_plant", "persistence_rmse",
           "exponential_decay_rmse", "HORIZONS_S"]

#: Horizons that matter.  60 s is a sanity check, 300 s is one MPC horizon, and
#: 1200 s is a full warm-up -- the case the whole project exists to control.
HORIZONS_S = (60, 300, 1200)


@dataclass
class BaselineScores:
    """Rollout RMSE in deg C at each horizon, per method."""

    plant: dict[int, float]
    persistence: dict[int, float]
    exponential: dict[int, float]

    def skill(self, horizon: int) -> float:
        """Fraction of the better baseline's error the plant removes.

        1.0 is perfect, 0.0 is no better than the baseline, negative is worse.
        """
        base = min(self.persistence[horizon], self.exponential[horizon])
        return 1.0 - self.plant[horizon] / base if base > 0 else float("nan")

    def table(self) -> str:
        rows = ["  horizon   plant   persist    expon    skill",
                "  " + "-" * 44]
        for h in sorted(self.plant):
            rows.append(f"  {h:>5}s  {self.plant[h]:6.3f}   {self.persistence[h]:6.3f}   "
                        f"{self.exponential[h]:6.3f}   {self.skill(h):+6.1%}")
        return "\n".join(rows)


@torch.no_grad()
def persistence_rmse(w: WindowSet, horizon: int) -> float:
    """Baseline: cabin temperature never changes from its initial value."""
    ws = w.truncate(horizon)
    pred = ws.y[:, :1].expand_as(ws.y)
    return float(torch.sqrt(torch.mean((pred - ws.y) ** 2)))


@torch.no_grad()
def exponential_decay_rmse(w: WindowSet, horizon: int, tau_s: float | None = None) -> float:
    """Baseline: first-order relaxation toward ambient, best-fit time constant.

    The tougher of the two baselines, and the honest one to beat: it already
    captures "a cabin tends toward the outside temperature", which is most of
    what a naive thermal model knows.  ``tau_s`` is fitted on the same windows
    it is scored on, deliberately giving the baseline every advantage.
    """
    ws = w.truncate(horizon)
    y = ws.y
    amb = ws.u[..., 2]
    t = torch.arange(ws.horizon, dtype=y.dtype, device=y.device).view(1, -1)

    candidates = torch.tensor([tau_s], dtype=y.dtype) if tau_s else torch.logspace(1.5, 4.5, 25)
    best = float("inf")
    for tau in candidates:
        decay = torch.exp(-t / tau)
        pred = amb + (y[:, :1] - amb) * decay
        best = min(best, float(torch.sqrt(torch.mean((pred - y) ** 2))))
    return best


def evaluate_plant(model: RCPlant, w: WindowSet,
                   horizons: tuple[int, ...] = HORIZONS_S) -> BaselineScores:
    """Rollout RMSE for the plant and both baselines at every horizon.

    The window's own full length is always scored, whatever ``horizons`` says.
    Silently dropping it would hide exactly the case the model is being built
    for: with 900 s windows and a 1200 s target horizon, filtering would report
    only 60 s and 300 s -- the two horizons where doing nothing is already a
    good answer -- and never show the long-horizon behaviour a controller
    depends on.
    """
    horizons = tuple(sorted({h for h in horizons if h <= w.horizon} | {w.horizon}))
    return BaselineScores(
        plant={h: rollout_rmse(model, w, h) for h in horizons},
        persistence={h: persistence_rmse(w, h) for h in horizons},
        exponential={h: exponential_decay_rmse(w, h) for h in horizons},
    )


@torch.no_grad()
def per_trip_rmse(model: RCPlant, w: WindowSet, horizon: int) -> dict[str, float]:
    """Rollout RMSE broken out by trip -- where a mean hides a failure."""
    ws = w.truncate(horizon)
    state0 = model.initial_state(ws.y[:, 0], ws.u[:, 0, 0])
    err = (model.rollout(state0, ws.u, ws.trip_idx)[..., 0] - ws.y) ** 2
    out = {}
    for i, name in enumerate(ws.trips):
        m = ws.trip_idx == i
        if m.any():
            out[name] = float(torch.sqrt(err[m].mean()))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def energy_balance_check(model: RCPlant, w: WindowSet) -> dict[str, float]:
    """Confirm the fitted model conserves energy over a rollout.

    Integrates every heat flow over a window and compares the net against the
    change in stored energy.  A mismatch means the integrator or the
    derivatives are wrong -- a class of bug that silently produces a
    plausible-looking fit and a controller that invents free heat.
    """
    with torch.no_grad():
        state0 = model.initial_state(w.y[:, 0], w.u[:, 0, 0])
        traj = model.rollout(state0, w.u, w.trip_idx)
        t_cab, t_mass, q_del = traj[..., 0], traj[..., 1], traj[..., 2]
        p_ac, amb, speed = w.u[..., 1], w.u[..., 2], w.u[..., 3]
        aux = model.aux_w(w.trip_idx).unsqueeze(-1)

        ua = model.ua0 + model.ua1 * speed
        flows = q_del - model.cop_ac * p_ac + aux + ua * (amb - t_cab)
        # trapezoidal integration over the window, in joules
        net = torch.trapezoid(flows, dx=1.0, dim=-1)
        stored = (model.c_cab * (t_cab[:, -1] - t_cab[:, 0])
                  + model.c_mass * (t_mass[:, -1] - t_mass[:, 0]))
        scale = torch.clamp(stored.abs(), min=1.0)
        rel = ((net - stored).abs() / scale)
        return {"max_rel_error": float(rel.max()), "mean_rel_error": float(rel.mean())}
