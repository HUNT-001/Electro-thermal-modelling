"""A differentiable grey-box thermal model of the cabin.

Why grey-box
------------
The controller in L3 needs something to optimise against, and whatever it
optimises against it will also *exploit*.  A black-box sequence model of cabin
temperature can be driven to physically impossible states by a planner
searching for cheap comfort, and nothing in its weights objects.  A network of
thermal capacitances and conductances cannot: energy that enters the cabin has
to come from somewhere, and a fitted capacitance that lands at 3 J/K is
visibly wrong in a way a fitted weight matrix never is.

So the model is physics first, with an optional bounded neural residual for
what the physics misses.

States
------
``T_cab``   cabin air temperature (measured -- this is what we fit)
``T_mass``  interior thermal mass: seats, trim, structure (latent)
``Q_del``   heat actually delivered to the cabin air (latent)

The third state matters more than it looks.  The heater warms coolant, the
coolant warms air in the heat exchanger, and the blower carries it into the
cabin.  Commanded electrical power therefore leads delivered heat by tens of
seconds.  A model without that lag has to explain the delay by inflating
``C_cab``, which then ruins the steady-state gain.

Dynamics
--------
.. math::

    \\dot{Q}_{del}  &= (\\eta\\,P_{heat} - Q_{del}) / \\tau_h \\\\
    C_{cab}\\dot{T}_{cab} &= Q_{del} - \\mathrm{COP}\\,P_{ac} + Q_{aux}
                            - UA_{amb}(v)\\,(T_{cab}-T_{amb})
                            - UA_{m}\\,(T_{cab}-T_{mass}) \\\\
    C_{m}\\dot{T}_{mass}  &= UA_{m}\\,(T_{cab}-T_{mass})

with a speed-dependent envelope conductance :math:`UA_{amb}(v)=UA_0+UA_1 v`,
since forced convection over the body scales with road speed.

The A/C term is not optional decoration.  The 32 summer trips run the
compressor essentially continuously -- fewer than 1 % of their rows have both
heater and A/C off -- so there is no passive regime hiding in this dataset to
identify :math:`C` and :math:`UA` from in isolation.  Modelling cooling as a
negative heat input is what makes those trips usable at all.

Parameterisation
----------------
Every physical parameter is stored as an unconstrained real and mapped through
``softplus`` (or ``sigmoid`` for the bounded ones), so gradient descent cannot
propose a negative capacitance.  Initial values are set from first-principles
estimates for a small BEV cabin, which also keeps the optimiser in a sane basin:

* a 7 kW heater takes roughly ten minutes to lift the cabin ~22 K
  :math:`\\Rightarrow C_{cab}\\sim10^5` J/K
* ~1.2 kW holds a 22 K rise at steady state
  :math:`\\Rightarrow UA_0\\sim55` W/K
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["PlantParams", "RCPlant", "PLAUSIBLE_RANGES"]

#: Index of each input channel in the ``u`` tensor passed to :meth:`RCPlant.rollout`.
INPUT_CHANNELS = ("p_heat_w", "p_ac_w", "amb_temp_c", "speed_ms")

#: Physically defensible ranges for a small BEV cabin.  Not constraints -- the
#: model is free to leave them -- but a fit that does has identified something
#: other than a car, and :func:`RCPlant.implausible` says so.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "c_cab": (2.0e4, 1.0e6),      # J/K
    "c_mass": (5.0e4, 5.0e6),     # J/K
    "ua0": (10.0, 300.0),         # W/K at standstill
    "ua1": (0.0, 40.0),           # W/K per m/s
    "ua_mass": (5.0, 500.0),      # W/K cabin air <-> interior mass
    "eta": (0.3, 1.0),            # electrical -> delivered heat
    "tau_h": (2.0, 300.0),        # s
    "cop_ac": (0.5, 6.0),         # cooling W per electrical W
}


def _inv_softplus(x: float) -> float:
    import math

    return math.log(math.expm1(x))


def _inv_sigmoid_scaled(x: float, lo: float, hi: float) -> float:
    import math

    p = (x - lo) / (hi - lo)
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass(frozen=True)
class PlantParams:
    """Identified parameters in physical units, for reporting and plausibility checks."""

    c_cab: float
    c_mass: float
    ua0: float
    ua1: float
    ua_mass: float
    eta: float
    tau_h: float
    cop_ac: float

    def implausible(self) -> dict[str, tuple[float, tuple[float, float]]]:
        """Parameters outside their physically defensible range."""
        bad = {}
        for name, (lo, hi) in PLAUSIBLE_RANGES.items():
            v = getattr(self, name)
            if not (lo <= v <= hi):
                bad[name] = (v, (lo, hi))
        return bad

    def steady_state_power_w(self, delta_t: float = 22.0, speed_ms: float = 0.0) -> float:
        """Heater power needed to hold ``delta_t`` above ambient at ``speed_ms``.

        A useful one-number sanity check: the measured trips settle around
        1.1--1.3 kW holding roughly 22 K in urban driving.
        """
        return (self.ua0 + self.ua1 * speed_ms) * delta_t / max(self.eta, 1e-6)

    def __str__(self) -> str:
        return (f"C_cab={self.c_cab:,.0f} J/K  C_mass={self.c_mass:,.0f} J/K  "
                f"UA0={self.ua0:.1f} W/K  UA1={self.ua1:.2f} W/K/(m/s)  "
                f"UA_mass={self.ua_mass:.1f} W/K  eta={self.eta:.3f}  "
                f"tau_h={self.tau_h:.1f} s  COP={self.cop_ac:.2f}")


class ResidualNet(nn.Module):
    """Small bounded correction to the cabin-air energy balance.

    Deliberately weak: two hidden layers, ``tanh`` output scaled to
    ``max_w`` watts.  The point is to absorb what the RC network cannot
    represent -- solar load, occupancy, recirculation flap changes -- without
    being able to overpower the physics.  If the residual is doing most of the
    work, the grey-box structure is wrong and that should be visible, not
    papered over.
    """

    def __init__(self, n_inputs: int = 6, hidden: int = 32, max_w: float = 500.0):
        super().__init__()
        self.max_w = max_w
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats: Tensor) -> Tensor:
        return self.max_w * torch.tanh(self.net(feats).squeeze(-1))


class RCPlant(nn.Module):
    """Three-state RC cabin model, integrated with fixed-step RK4.

    Parameters
    ----------
    n_trips:
        Number of trips that will be fitted.  Each gets its own scalar
        auxiliary heat term ``Q_aux`` (solar gain, occupants, recirculation
        setting) -- effects that are real, constant-ish within a trip, and not
        measured.  Bounded to ``aux_max_w`` so it cannot silently become a
        second heater and absorb the model's errors.
    use_residual:
        Add the bounded neural correction term.
    """

    def __init__(self, n_trips: int = 1, use_residual: bool = False,
                 aux_max_w: float = 400.0, residual_max_w: float = 500.0):
        super().__init__()
        self.aux_max_w = aux_max_w

        self._c_cab = nn.Parameter(torch.tensor(_inv_softplus(1.0e5 / 1e4)))
        self._c_mass = nn.Parameter(torch.tensor(_inv_softplus(4.0e5 / 1e4)))
        self._ua0 = nn.Parameter(torch.tensor(_inv_softplus(55.0)))
        self._ua1 = nn.Parameter(torch.tensor(_inv_softplus(2.0)))
        self._ua_mass = nn.Parameter(torch.tensor(_inv_softplus(60.0)))
        self._eta = nn.Parameter(torch.tensor(_inv_sigmoid_scaled(0.85, *PLAUSIBLE_RANGES["eta"])))
        self._tau_h = nn.Parameter(torch.tensor(_inv_softplus(30.0 - 1.0)))
        self._cop_ac = nn.Parameter(torch.tensor(_inv_softplus(2.0)))
        self._aux = nn.Parameter(torch.zeros(n_trips))

        self.residual = ResidualNet(max_w=residual_max_w) if use_residual else None

    # -- parameter accessors (physical units) ------------------------------
    @property
    def c_cab(self) -> Tensor:
        return nn.functional.softplus(self._c_cab) * 1e4

    @property
    def c_mass(self) -> Tensor:
        return nn.functional.softplus(self._c_mass) * 1e4

    @property
    def ua0(self) -> Tensor:
        return nn.functional.softplus(self._ua0)

    @property
    def ua1(self) -> Tensor:
        return nn.functional.softplus(self._ua1)

    @property
    def ua_mass(self) -> Tensor:
        return nn.functional.softplus(self._ua_mass)

    @property
    def eta(self) -> Tensor:
        lo, hi = PLAUSIBLE_RANGES["eta"]
        return lo + (hi - lo) * torch.sigmoid(self._eta)

    @property
    def tau_h(self) -> Tensor:
        return nn.functional.softplus(self._tau_h) + 1.0

    @property
    def cop_ac(self) -> Tensor:
        return nn.functional.softplus(self._cop_ac)

    def aux_w(self, trip_idx: Tensor) -> Tensor:
        return self.aux_max_w * torch.tanh(self._aux[trip_idx])

    #: Values the parameters start at, from first-principles estimates.
    INITIAL = dict(c_cab=1.0e5, c_mass=4.0e5, ua0=55.0, ua1=2.0,
                   ua_mass=60.0, eta=0.85, tau_h=30.0, cop_ac=2.0)

    def movement_from_initial(self) -> dict[str, float]:
        """Relative distance each parameter has travelled from its starting value.

        A necessary check on any identification run, and one that is easy to
        skip because the alternative looks like success.  Parameters that agree
        to a fraction of a percent across folds fitted on *different* data are
        not necessarily well identified -- they may simply not have moved, and
        a fit that is still sitting on its initialisation will report beautiful
        cross-fold stability while having learned nothing.  Movement near zero
        on a parameter the data should constrain means undertrained, an
        unexcited input, or a vanishing gradient -- not convergence.
        """
        got = self.params()
        return {k: abs(getattr(got, k) - v) / v for k, v in self.INITIAL.items()}

    def params(self) -> PlantParams:
        """Snapshot the identified parameters in physical units."""
        with torch.no_grad():
            return PlantParams(
                c_cab=float(self.c_cab), c_mass=float(self.c_mass),
                ua0=float(self.ua0), ua1=float(self.ua1),
                ua_mass=float(self.ua_mass), eta=float(self.eta),
                tau_h=float(self.tau_h), cop_ac=float(self.cop_ac),
            )

    # -- dynamics ----------------------------------------------------------
    def derivatives(self, state: Tensor, u: Tensor, aux: Tensor) -> Tensor:
        """Continuous-time derivatives.

        ``state``: ``(..., 3)`` of ``[T_cab, T_mass, Q_del]``.
        ``u``:     ``(..., 4)`` of :data:`INPUT_CHANNELS`.
        ``aux``:   ``(...,)`` auxiliary heat in watts.
        """
        t_cab, t_mass, q_del = state.unbind(-1)
        p_heat, p_ac, t_amb, speed = u.unbind(-1)

        ua_amb = self.ua0 + self.ua1 * speed
        q_env = ua_amb * (t_amb - t_cab)
        q_mass = self.ua_mass * (t_mass - t_cab)
        q_cool = self.cop_ac * p_ac

        q_in = q_del - q_cool + aux
        if self.residual is not None:
            feats = torch.stack(
                [t_cab / 40.0, t_mass / 40.0, q_del / 5000.0,
                 p_heat / 7000.0, t_amb / 40.0, speed / 30.0], dim=-1)
            q_in = q_in + self.residual(feats)

        d_t_cab = (q_in + q_env + q_mass) / self.c_cab
        d_t_mass = -q_mass / self.c_mass
        d_q_del = (self.eta * p_heat - q_del) / self.tau_h
        return torch.stack([d_t_cab, d_t_mass, d_q_del], dim=-1)

    def step(self, state: Tensor, u0: Tensor, u1: Tensor, aux: Tensor, dt: float) -> Tensor:
        """One classical RK4 step.

        Inputs are held over the interval by linear interpolation between the
        samples at each end (``u0``, ``u1``), which is what the RK4 midpoint
        stages need and what a zero-order hold would get wrong on a fast heater
        ramp.
        """
        u_mid = 0.5 * (u0 + u1)
        k1 = self.derivatives(state, u0, aux)
        k2 = self.derivatives(state + 0.5 * dt * k1, u_mid, aux)
        k3 = self.derivatives(state + 0.5 * dt * k2, u_mid, aux)
        k4 = self.derivatives(state + dt * k3, u1, aux)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, state0: Tensor, u: Tensor, trip_idx: Tensor, dt: float = 1.0) -> Tensor:
        """Free-running simulation over a whole window.

        ``state0``: ``(B, 3)`` initial state.
        ``u``:      ``(B, T, 4)`` input sequence.
        Returns ``(B, T, 3)`` -- the trajectory including the initial state.

        There is no teacher forcing anywhere in here.  The model sees the true
        cabin temperature exactly once, at ``t=0``, and must carry itself for
        the rest of the window.  One-step prediction of a signal with a
        multi-minute time constant is trivially accurate and says nothing about
        whether the dynamics are right; this is the honest test.
        """
        aux = self.aux_w(trip_idx)
        states = [state0]
        s = state0
        for t in range(u.shape[1] - 1):
            s = self.step(s, u[:, t], u[:, t + 1], aux, dt)
            states.append(s)
        return torch.stack(states, dim=1)

    def initial_state(self, t_cab0: Tensor, p_heat0: Tensor) -> Tensor:
        """Build a state vector from what is actually observable at ``t=0``.

        ``T_mass`` is latent and initialised to the cabin temperature.  That is
        exactly right on a cold start, where cabin and interior have both been
        soaking at ambient, and increasingly wrong on a warm restart -- which is
        one reason the three trips logged "directly after previous trip"
        (TripB16, B17, B28) are kept in their own session group rather than
        split across train and test.
        """
        q_del0 = self.eta.detach() * p_heat0
        return torch.stack([t_cab0, t_cab0, q_del0], dim=-1)
