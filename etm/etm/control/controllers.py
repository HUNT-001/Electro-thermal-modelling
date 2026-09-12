"""Controllers, from the ones that must be beaten to the one being proposed.

An MPC result means nothing without the ladder underneath it.  ``OEMReplay`` is
the shipped BMW policy as recorded; a thermostat and a tuned PI are what any
competent engineer would build in an afternoon.  If model-predictive control
cannot beat those it has not earned its complexity, and reporting it without
them would be the same mistake as reporting a single train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .cost import ComfortSpec
from .env import HEATER_MAX_W, ThermalEnv

__all__ = ["OEMReplay", "Thermostat", "PIController", "MPCController"]


class OEMReplay:
    """Play back the heater trace the production controller actually commanded.

    The baseline the whole project is measured against, and the reason this
    dataset is worth so much: the comparison is against a real shipped
    controller on the same trip, not against a strawman.
    """

    name = "oem"

    def __call__(self, k: int, t_cab: float, env: ThermalEnv) -> float:
        s = env.scenario
        return float(s.p_heat_oem[min(k, len(s) - 1)])


@dataclass
class Thermostat:
    """Bang-bang: full power below the band, off above it."""

    band_c: float = 1.0
    setpoint_c: float = 22.0
    power_w: float = HEATER_MAX_W
    name: str = "thermostat"

    def __call__(self, k: int, t_cab: float, env: ThermalEnv) -> float:
        return self.power_w if t_cab < self.setpoint_c - self.band_c else 0.0


@dataclass
class PIController:
    """Proportional-integral on cabin temperature error, with anti-windup.

    The honest engineering baseline.  Gains are in watts per kelvin; the
    integral term is clamped so a long cold soak cannot wind up and overshoot
    once the cabin finally responds.
    """

    kp: float = 800.0
    ki: float = 4.0
    setpoint_c: float = 22.0
    i_max_w: float = 4000.0
    name: str = "pi"

    def __post_init__(self):
        self._i = 0.0

    def reset(self):
        self._i = 0.0

    def __call__(self, k: int, t_cab: float, env: ThermalEnv) -> float:
        if k == 0:
            self._i = 0.0
        e = self.setpoint_c - t_cab
        self._i = float(np.clip(self._i + self.ki * e * env.scenario.dt_s,
                                -self.i_max_w, self.i_max_w))
        return float(np.clip(self.kp * e + self._i, 0.0, HEATER_MAX_W))


@dataclass
class MPCController:
    """Receding-horizon control by gradient descent through the learned plant.

    The plant is differentiable, so the heater schedule over the horizon can be
    optimised directly: simulate forward, backpropagate the cost to the planned
    actions, step, repeat.  No sampling, no surrogate.

    Two choices keep it tractable and honest.

    **Piecewise-constant actions.**  The schedule is parameterised as
    ``n_blocks`` constant segments rather than one variable per timestep.  With
    a 44 s heater lag the plant cannot follow anything faster anyway, and it cuts
    the decision vector from hundreds of variables to a handful -- which is also
    what makes the whole evaluation finish in minutes rather than overnight.

    **Re-planning on a coarser clock than simulation.**  The optimiser runs
    every ``replan_every`` steps and the first block is held until the next
    re-plan, exactly as a real supervisory controller would.

    ``comfort_weight`` is the single knob traded off against energy.  Sweeping
    it is what traces the frontier that :mod:`etm.control.evaluate` compares
    against the OEM point -- a controller is not one operating point, it is a
    curve, and comparing single points would let either side win by choosing a
    different comfort target.
    """

    comfort_weight: float = 1.0
    horizon_s: float = 600.0
    n_blocks: int = 6
    replan_every: int = 6
    iters: int = 40
    lr: float = 900.0
    spec: ComfortSpec = None
    switch_weight: float = 0.0
    name: str = "mpc"

    def __post_init__(self):
        self.spec = self.spec or ComfortSpec()
        self._plan: np.ndarray | None = None
        self._plan_k: int = -10 ** 9
        self._state: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None

    def reset(self):
        self._plan = None
        self._plan_k = -10 ** 9
        self._state = None
        self._logits = None

    def _plan_cost(self, blocks: torch.Tensor, state: torch.Tensor,
                   env: ThermalEnv, k0: int, steps_per_block: int) -> torch.Tensor:
        """Simulate the planned schedule and return its scalar cost."""
        s = env.scenario
        aux = torch.zeros(1)
        cost = torch.zeros(())
        cur = state
        p_prev = None
        for b in range(len(blocks)):
            # squash to the actuator range: the optimiser is unconstrained, the
            # heater is not, and a plan that commands 12 kW is not a plan
            p = HEATER_MAX_W * torch.sigmoid(blocks[b])
            for j in range(steps_per_block):
                k = k0 + b * steps_per_block + j
                u0 = env.inputs_at(k, p)
                u1 = env.inputs_at(k + 1, p)
                cur = env.plant.step(cur, u0, u1, aux, s.dt_s)
                t_cab = cur[0, 0]
                cold = torch.clamp(self.spec.setpoint_c - self.spec.band_c - t_cab, min=0.0)
                warm = torch.clamp(t_cab - (self.spec.setpoint_c + self.spec.band_c), min=0.0)
                discomfort = self.spec.cold_weight * cold + warm
                # energy in Wh, discomfort in K-min: the same units the result
                # is reported in, so comfort_weight is interpretable as Wh per
                # K-minute rather than an arbitrary scale factor
                cost = cost + p * s.dt_s / 3600.0
                cost = cost + self.comfort_weight * discomfort * s.dt_s / 60.0
            if p_prev is not None and self.switch_weight:
                cost = cost + self.switch_weight * torch.abs(p - p_prev) / 1000.0
            p_prev = p
        return cost

    def __call__(self, k: int, t_cab: float, env: ThermalEnv) -> float:
        if k == 0:
            self.reset()
        steps_per_block = max(1, int(round(self.horizon_s / env.scenario.dt_s / self.n_blocks)))

        if self._plan is None or k - self._plan_k >= self.replan_every:
            # the controller knows the measured cabin temperature; the interior
            # mass and delivered heat it carries forward from its own model,
            # which is what a real supervisory controller with a state observer
            # would have
            if getattr(self, "_state", None) is None:
                state = torch.tensor([[t_cab, t_cab, 0.0]], dtype=torch.float32)
            else:
                state = self._state.clone()
                state[0, 0] = t_cab

            # the closed-loop rollout runs under no_grad; planning needs the
            # graph back, so re-enable it just for the optimisation
            with torch.enable_grad():
                # Warm start from the previous plan, shifted one block forward.
                # A receding-horizon controller re-solves a problem it has
                # mostly already solved; starting from zero every time both
                # wastes iterations and leaves the solution short of converged,
                # which shows up as a controller that will not fully commit to
                # switching the heater off.
                if self._logits is None:
                    init = torch.full((self.n_blocks,), -2.0)     # ~840 W, low but not zero
                else:
                    init = torch.cat([self._logits[1:], self._logits[-1:]]).clone()
                blocks = init.detach().requires_grad_(True)
                opt = torch.optim.Adam([blocks], lr=0.35)
                for _ in range(self.iters):
                    opt.zero_grad()
                    c = self._plan_cost(blocks, state.detach(), env, k, steps_per_block)
                    c.backward()
                    opt.step()
                with torch.no_grad():
                    self._logits = blocks.detach().clone()
                    self._plan = (HEATER_MAX_W * torch.sigmoid(blocks)).numpy()
            self._plan_k = k

        idx = min((k - self._plan_k) // steps_per_block, len(self._plan) - 1)
        p = float(self._plan[idx])

        # carry the model's own state forward so the next re-plan starts from
        # a consistent latent estimate rather than re-guessing it
        with torch.no_grad():
            if getattr(self, "_state", None) is None:
                self._state = torch.tensor([[t_cab, t_cab, 0.0]], dtype=torch.float32)
            st = self._state.clone()
            st[0, 0] = t_cab
            u0 = env.inputs_at(k, p)
            u1 = env.inputs_at(k + 1, p)
            self._state = env.plant.step(st, u0, u1, torch.zeros(1), env.scenario.dt_s)
        return p
