"""A deterministic safety layer between the learned controller and the heater.

Everything above this file is a model: an identified plant, a forecaster, an
optimiser.  None of it is something a vehicle-safety engineer can certify,
because none of it is guaranteed to do anything in particular.  This file is the
part that can be certified, and it is deliberately the least clever code in the
project: no learning, no optimisation, a handful of inspectable rules with final
say over what current the heater actually draws.

The contract is simple.  The learned controller *proposes*; the supervisor
*disposes*.  On every step the proposed action passes through an ordered set of
guards.  If any guard trips -- the action is out of range, changing too fast, or
the vehicle has left the envelope the plant was identified on, or a sensor has
dropped out -- the supervisor overrides it, and on a genuine fault it hands
control to a deterministic fallback (the production controller, or a PI loop)
rather than trusting a model outside the conditions it was built for.

Every override is recorded with its reason, so the takeover is auditable after
the fact -- which is exactly what an OEM evaluating this would ask for: not
"does the ML controller work" but "what happens when it doesn't, and can you
prove the car stays safe".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import HEATER_MAX_W, ThermalEnv

__all__ = ["Guard", "ActuatorClamp", "RateLimit", "InputEnvelope", "SensorValid",
           "SupervisedController", "Takeover", "SupervisorReport"]


@dataclass
class Takeover:
    """A single step on which the supervisor changed or replaced the action."""

    k: int
    reason: str
    proposed_w: float
    applied_w: float
    fell_back: bool


class Guard:
    """Base class: inspect (and optionally amend) a proposed action.

    A guard returns ``(action, reason_or_None, fault)``.  ``reason`` is set when
    the guard changed the action; ``fault`` is True when the situation is one
    the learned controller should not be trusted in at all, which triggers the
    deterministic fallback for this step.
    """

    def check(self, k: int, proposed_w: float, t_cab: float,
              env: ThermalEnv) -> tuple[float, str | None, bool]:
        raise NotImplementedError


@dataclass
class ActuatorClamp(Guard):
    """Hard physical limit: the heater cannot draw outside ``[0, max_w]``.

    Non-negotiable and non-faulting -- a proposed 12 kW is clamped to the rating
    every time, because the hardware would clamp it anyway; the point is that the
    software does it first, visibly, rather than sending an impossible command.
    """

    max_w: float = HEATER_MAX_W

    def check(self, k, proposed_w, t_cab, env):
        if not np.isfinite(proposed_w):
            return 0.0, "non-finite action", True          # a NaN is a fault, not a clamp
        clamped = float(np.clip(proposed_w, 0.0, self.max_w))
        reason = None if clamped == proposed_w else "actuator limit"
        return clamped, reason, False


@dataclass
class RateLimit(Guard):
    """Bound how fast the heater command may change, in W per second.

    Protects the coolant loop and the electrical system from step commands a
    real actuator could not follow, and stops an optimiser from chattering.  Not
    a fault: the command is simply slew-limited toward the request.
    """

    max_rate_w_per_s: float = 2000.0
    _prev: float | None = field(default=None, repr=False)

    def reset(self):
        self._prev = None

    def check(self, k, proposed_w, t_cab, env):
        if k == 0:
            self._prev = None
        if self._prev is None:
            self._prev = proposed_w
            return proposed_w, None, False
        max_step = self.max_rate_w_per_s * env.scenario.dt_s
        applied = float(np.clip(proposed_w, self._prev - max_step, self._prev + max_step))
        self._prev = applied
        reason = None if abs(applied - proposed_w) < 1e-6 else "rate limit"
        return applied, reason, False


@dataclass
class InputEnvelope(Guard):
    """Flag operation outside the envelope the plant was identified on.

    The plant was fitted on winter trips with ambient in roughly [-5, 12] degC
    and road speed up to ~42 m/s.  Outside that, the model is extrapolating and
    its optimiser should not be trusted -- so this is a *fault* that engages the
    fallback, not a silent clamp.  The bounds are generous on purpose: the guard
    is meant to catch genuinely out-of-distribution operation (a heatwave, an
    autobahn blast well beyond the data), not to fire on ordinary variation.
    """

    amb_lo_c: float = -10.0
    amb_hi_c: float = 15.0
    speed_max_ms: float = 45.0

    def check(self, k, proposed_w, t_cab, env):
        s = env.scenario
        j = min(k, len(s) - 1)
        amb, speed = float(s.amb_c[j]), float(s.speed_ms[j])
        if not (self.amb_lo_c <= amb <= self.amb_hi_c):
            return proposed_w, f"ambient {amb:.0f}C outside identified envelope", True
        if speed > self.speed_max_ms:
            return proposed_w, f"speed {speed:.0f} m/s beyond identified envelope", True
        return proposed_w, None, False


@dataclass
class SensorValid(Guard):
    """Fault on a missing or physically impossible cabin temperature.

    A control loop reading a NaN or a stuck/absurd sensor value must not act on
    it.  The supervisor's model-free fallback (a PI loop, or the OEM trace) is
    the safe thing to run until the signal is sane again.
    """

    cab_lo_c: float = -40.0
    cab_hi_c: float = 60.0

    def check(self, k, proposed_w, t_cab, env):
        try:
            ok = np.isfinite(t_cab) and (self.cab_lo_c <= t_cab <= self.cab_hi_c)
        except TypeError:
            ok = False                                  # a non-numeric reading is a fault
        if not ok:
            return proposed_w, f"implausible cabin sensor ({t_cab})", True
        return proposed_w, None, False


@dataclass
class SupervisorReport:
    """What the supervisor did over a trip -- the audit trail."""

    n_steps: int
    takeovers: list[Takeover] = field(default_factory=list)

    @property
    def n_fallback(self) -> int:
        return sum(t.fell_back for t in self.takeovers)

    @property
    def n_amended(self) -> int:
        return sum(not t.fell_back for t in self.takeovers)

    def reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.takeovers:
            out[t.reason] = out.get(t.reason, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def summary(self) -> str:
        parts = [f"supervisor: {self.n_amended} actions amended, "
                 f"{self.n_fallback} steps on fallback, of {self.n_steps}"]
        for reason, n in self.reasons().items():
            parts.append(f"    {n:>5}  {reason}")
        return "\n".join(parts)


@dataclass
class SupervisedController:
    """Wrap a learned controller in the deterministic safety layer.

    ``primary`` proposes; each guard in order may amend the action or declare a
    fault; on a fault the ``fallback`` policy's action is used for that step.
    Guards run *after* the fallback decision too, so even the fallback's command
    is clamped and rate-limited -- the actuator limits hold no matter who is
    driving.
    """

    primary: object
    fallback: object
    guards: list[Guard] = field(default_factory=list)
    name: str = "supervised"
    report: SupervisorReport | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.guards:
            self.guards = [SensorValid(), InputEnvelope(), ActuatorClamp(), RateLimit()]

    def reset(self):
        for c in (self.primary, self.fallback, *self.guards):
            if hasattr(c, "reset"):
                c.reset()
        self.report = None

    def __call__(self, k: int, t_cab: float, env: ThermalEnv) -> float:
        if k == 0:
            self.reset()
            self.report = SupervisorReport(n_steps=len(env.scenario))

        # 1. does any guard see a fault in the *current situation*?  (sensor /
        #    envelope faults do not depend on the proposed action, so decide the
        #    source of the command before computing it)
        fault_reason = None
        for g in self.guards:
            _, reason, fault = g.check(k, 0.0, t_cab, env)
            if fault:
                fault_reason = reason
                break

        fell_back = fault_reason is not None
        source = self.fallback if fell_back else self.primary
        proposed = source(k, t_cab, env)

        # 2. clamp / slew every command, whoever produced it
        applied = proposed
        amend_reason = None
        for g in self.guards:
            applied, reason, fault = g.check(k, applied, t_cab, env)
            if reason and amend_reason is None:
                amend_reason = reason
            if fault and not fell_back:
                # the applied action itself is faulty (e.g. NaN from the primary)
                fell_back = True
                fault_reason = reason
                applied = self.fallback(k, t_cab, env)
                applied = float(np.clip(applied, 0.0, ActuatorClamp().max_w))

        if fell_back or amend_reason:
            self.report.takeovers.append(Takeover(
                k=k, reason=fault_reason or amend_reason,
                proposed_w=float(proposed) if np.isfinite(proposed) else float("nan"),
                applied_w=float(applied), fell_back=fell_back))
        return float(applied)
