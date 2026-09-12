"""Identify the plant parameters by minimising multi-step rollout error.

Two things here are deliberate and worth not undoing.

**The loss is on free-running rollouts, never one-step prediction.**  Cabin
temperature at 1 Hz has a multi-minute time constant, so ``T(t+1) ~= T(t)``
scores brilliantly and proves nothing.  A model fitted on one-step error learns
the identity function and falls apart the moment a controller asks it to
predict twenty minutes ahead -- which is the only thing it will ever be asked
to do.

**Horizon curriculum.**  Optimising a 1200-step rollout from a cold start is a
deep recurrence with an ill-conditioned loss surface; gradients either vanish or
explode and the fit stalls at a local minimum where ``C_cab`` has absorbed the
heater lag.  Fitting short windows first pins down the fast dynamics
(:math:`\\tau_h`, ``C_cab``), then lengthening the horizon identifies the slow
ones (``C_mass``, ``UA``) without disturbing them.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .model import PLAUSIBLE_RANGES, RCPlant
from .windows import WindowSet

log = logging.getLogger(__name__)

__all__ = ["FitConfig", "FitResult", "fit_plant", "rollout_loss",
           "plausibility_penalty", "estimate_runtime"]


@dataclass
class FitConfig:
    """Knobs for identification.  Defaults are tuned for ~35 winter trips at 1 Hz."""

    horizons_s: tuple[int, ...] = (120, 300, 600, 1200)
    #: Epochs per curriculum stage.  An int applies to every stage; a tuple
    #: gives one count per horizon.  Cost scales with horizon x epochs, so the
    #: 1200 s stage alone is ten times the cost of the 120 s stage at equal
    #: epoch counts -- spend the epochs where the fast dynamics are learned and
    #: taper them as the horizon grows.
    epochs_per_stage: int | tuple[int, ...] = 60
    batch_size: int = 32
    lr: float = 3e-2
    lr_residual: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    huber_delta: float = 1.0      # deg C; robust to sensor spikes
    #: Weight on the delivery-air temperature term.  Non-zero is what separates
    #: eta from C_cab; too large and the duct sensor's own dynamics start
    #: dictating the cabin capacitance.
    vent_weight: float = 0.3
    #: Soft penalty pulling parameters back inside their plausible ranges.
    #: Reporting implausibility after the fact is not enough -- the winter run
    #: drove the interior mass to a 26 s time constant in all five folds and
    #: still reduced the loss.
    plausibility_weight: float = 0.05
    device: str = "cpu"
    seed: int = 0
    log_every: int = 20
    #: Print a progress line while fitting.  On by default: a run that prints
    #: nothing for its first half hour is indistinguishable from a hung one.
    progress: bool = True

    def epochs_for(self, horizon_s: int) -> int:
        if isinstance(self.epochs_per_stage, int):
            return self.epochs_per_stage
        i = self.horizons_s.index(horizon_s)
        return self.epochs_per_stage[i]


@dataclass
class FitResult:
    model: RCPlant
    history: list[dict] = field(default_factory=list)
    val_rmse_by_horizon: dict[int, float] = field(default_factory=dict)
    #: Parameters held at their initial value because nothing in the data
    #: could constrain them.
    frozen: list[str] = field(default_factory=list)


def unidentifiable(w: WindowSet, min_active_frac: float = 0.02) -> set[str]:
    """Parameters whose input is never meaningfully exercised in these windows.

    The A/C compressor runs in 0-6 % of winter rows and averages a few watts,
    so ``cop_ac`` has almost no gradient -- and a 660-epoch run duly drifted it
    to 6.0-7.3, above any physical COP, while the loss barely noticed.  A
    number fitted with no information in it is worse than no number: it looks
    like a result.  Freezing it and saying so is the honest option.
    """
    out: set[str] = set()
    if float((w.u[..., 1].abs() > 50.0).float().mean()) < min_active_frac:
        out.add("cop_ac")
    # UA1 is the speed dependence of envelope loss, and on this dataset it is
    # not merely unexcited but genuinely ~zero: a steady-state heat-balance
    # regression across all 38 winter trips (see etm.plant.steady_state) puts it
    # at +0.007 W/K per m/s with per-trip fixed effects, and the within-trip
    # correlation between speed and loss is ~0.05.  The i3 recirculates cabin
    # air once warm, so envelope loss is conduction-dominated and speed-
    # insensitive.  Freeze UA1 unless a window actually pairs sustained high
    # speed with a warm cabin -- which no trip_start window does -- and let the
    # `etm identify-ua` diagnostic be the record of why.
    per_window_hi = (w.u[..., 3] > 20.0).float().mean(dim=1)     # frac above 72 km/h
    if float(per_window_hi.max()) < 0.15:
        out.add("ua1")
    return out


def _simulate(model: RCPlant, w: WindowSet):
    """Roll the model over the scored part of every window.

    With a burn-in the latent states are estimated from the measured prefix
    (see :meth:`RCPlant.initial_state_from_history`) and the rollout starts at
    the end of it; without one they fall back to the cold-start guess.
    """
    b = w.burn_in
    if b:
        state0 = model.initial_state_from_history(w.y[:, :b], w.u[:, :b])
    else:
        state0 = model.initial_state(w.y[:, 0], w.u[:, 0, 0])
    traj = model.rollout(state0, w.u[:, b:], w.trip_idx)
    return traj, w.y[:, b:], None if w.y_vent is None else w.y_vent[:, b:]


def plausibility_penalty(model: RCPlant) -> Tensor:
    """Smooth hinge pulling every parameter back inside its plausible range.

    Scaled in log space so a capacitance and a time constant are penalised
    comparably.  Zero inside the range, so a well-behaved fit pays nothing.
    """
    total = torch.zeros((), device=model._c_cab.device)
    for name, (lo, hi) in PLAUSIBLE_RANGES.items():
        v = getattr(model, name)
        lo_t = torch.as_tensor(lo, device=v.device, dtype=v.dtype)
        hi_t = torch.as_tensor(hi, device=v.device, dtype=v.dtype)
        below = torch.clamp(torch.log(lo_t) - torch.log(v.clamp(min=1e-9)), min=0.0)
        above = torch.clamp(torch.log(v.clamp(min=1e-9)) - torch.log(hi_t), min=0.0)
        total = total + below ** 2 + above ** 2
    return total


def rollout_loss(model: RCPlant, w: WindowSet, delta: float = 1.0,
                 vent_weight: float = 0.0, plausibility_weight: float = 0.0) -> Tensor:
    """Multi-output rollout loss: cabin temperature, plus the delivery air.

    Huber rather than MSE because the cabin sensor produces occasional single
    sample spikes; squared error lets one such sample dominate a whole window's
    gradient and drag the fitted capacitance with it.

    When the window set carries a duct temperature it is scored too.  That
    second output is not a refinement -- without it only ``eta / C_cab`` is
    identifiable and the fit is free to pick any point along that ridge.
    """
    traj, y, y_vent = _simulate(model, w)
    loss = torch.nn.functional.huber_loss(traj[..., 0], y, delta=delta)
    if vent_weight and y_vent is not None:
        loss = loss + vent_weight * torch.nn.functional.huber_loss(
            model.vent_temp(traj), y_vent, delta=delta)
    if plausibility_weight:
        loss = loss + plausibility_weight * plausibility_penalty(model)
    return loss


@torch.no_grad()
def rollout_rmse(model: RCPlant, w: WindowSet, horizon: int | None = None) -> float:
    """Open-loop RMSE (deg C) of cabin temperature over the window."""
    ws = w.truncate(horizon) if horizon else w
    traj, y, _ = _simulate(model, ws)
    return float(torch.sqrt(torch.mean((traj[..., 0] - y) ** 2)))


def fit_plant(
    train: WindowSet,
    val: WindowSet | None = None,
    cfg: FitConfig | None = None,
    use_residual: bool = False,
) -> FitResult:
    """Identify plant parameters from ``train`` windows.

    The residual network, when enabled, is trained at a much lower learning
    rate than the physical parameters.  If both moved at the same speed the
    network -- which has hundreds of parameters against the physics' eight --
    would win every gradient step, fit the data first, and leave the physical
    parameters at their initialisation.  The result would look accurate and
    identify nothing.
    """
    cfg = cfg or FitConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = RCPlant(n_trips=len(train.trips), use_residual=use_residual).to(device)
    train = train.to(device)
    val = val.to(device) if val is not None else None

    frozen = unidentifiable(train)
    for name in frozen:
        getattr(model, f"_{name}").requires_grad_(False)
    if frozen and cfg.progress:
        print(f"    frozen (input never excited in this fold): {', '.join(sorted(frozen))}",
              file=sys.stderr, flush=True)
    result_frozen = frozen

    physical = [p for n, p in model.named_parameters()
                if not n.startswith("residual") and p.requires_grad]
    groups = [{"params": physical, "lr": cfg.lr}]
    if model.residual is not None:
        groups.append({"params": model.residual.parameters(), "lr": cfg.lr_residual})
    opt = torch.optim.Adam(groups, weight_decay=cfg.weight_decay)

    result = FitResult(model=model, frozen=sorted(result_frozen))
    n = len(train)
    n_batches = max(1, -(-n // cfg.batch_size))
    total_units = sum(min(h, train.horizon) * cfg.epochs_for(h) * n_batches
                      for h in cfg.horizons_s)
    done_units = 0
    t_start = time.perf_counter()

    for horizon_s in cfg.horizons_s:
        stage = train.truncate(horizon_s)
        if stage.horizon < 2:
            continue
        n_epochs = cfg.epochs_for(horizon_s)
        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=device)
            total, nb = 0.0, 0
            for i in range(0, n, cfg.batch_size):
                batch = stage.subset(perm[i:i + cfg.batch_size].cpu())
                opt.zero_grad(set_to_none=True)
                loss = rollout_loss(model, batch, cfg.huber_delta,
                                    cfg.vent_weight, cfg.plausibility_weight)
                if not torch.isfinite(loss):
                    log.warning("non-finite loss at horizon %ds epoch %d; skipping batch",
                                horizon_s, epoch)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                total += float(loss.detach())
                nb += 1
                done_units += stage.horizon
            if cfg.progress and nb:
                elapsed = time.perf_counter() - t_start
                frac = done_units / max(total_units, 1)
                remain = elapsed / frac - elapsed if frac > 0 else 0.0
                print(f"\r    [{horizon_s:>4}s] epoch {epoch + 1:>4}/{n_epochs}  "
                      f"loss={total / nb:8.4f}  {100 * frac:5.1f}%  "
                      f"elapsed {elapsed / 60:5.1f}m  eta {remain / 60:5.1f}m",
                      end="", file=sys.stderr, flush=True)
            if nb and (epoch % cfg.log_every == 0 or epoch == n_epochs - 1):
                rec = {"horizon_s": horizon_s, "epoch": epoch, "train_loss": total / nb}
                if val is not None:
                    rec["val_rmse"] = rollout_rmse(model, val, horizon_s)
                result.history.append(rec)
                log.info("h=%4ds ep=%3d loss=%.4f%s", horizon_s, epoch, total / nb,
                         f" val_rmse={rec['val_rmse']:.3f}C" if val is not None else "")

    if cfg.progress:
        print(file=sys.stderr, flush=True)

    if val is not None:
        for h in (60, 300, 1200):
            if h <= val.horizon:
                result.val_rmse_by_horizon[h] = rollout_rmse(model, val, h)
    return result


def estimate_runtime(train: WindowSet, cfg: FitConfig, n_folds: int = 1,
                     use_residual: bool = False) -> float:
    """Time one optimiser step per stage and project the whole run, in seconds.

    Worth knowing before committing: cost is linear in horizon x epochs, so the
    1200 s stage dominates, and the sequential RK4 loop over a tiny state means
    a GPU is launch-bound and often no faster than the CPU here.
    """
    model = RCPlant(n_trips=len(train.trips), use_residual=use_residual).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    probe = train.subset(torch.arange(min(cfg.batch_size, len(train)))).to(cfg.device)
    n_batches = max(1, -(-len(train) // cfg.batch_size))

    total = 0.0
    for h in cfg.horizons_s:
        stage = probe.truncate(h)
        if stage.horizon < 2:
            continue
        opt.zero_grad(set_to_none=True)
        rollout_loss(model, stage).backward()      # warm up allocator / autograd
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        rollout_loss(model, stage).backward()
        opt.step()
        total += (time.perf_counter() - t0) * n_batches * cfg.epochs_for(h)
    return total * n_folds


def transfer_aux(model: RCPlant, n_trips: int) -> RCPlant:
    """Re-issue a fitted model for a different trip set, keeping the physics.

    The per-trip auxiliary heat terms are properties of individual trips (solar
    load, passengers) and cannot transfer to unseen ones.  Evaluating on a
    held-out trip therefore uses ``Q_aux = 0``, which is the honest setting: at
    deployment there is no fitted term for a trip that has not happened yet.
    """
    fresh = RCPlant(n_trips=n_trips, use_residual=model.residual is not None)
    with torch.no_grad():
        for name in ("_c_cab", "_tau_mass", "_ua0", "_ua1", "_ua_mass",
                     "_eta", "_tau_h", "_cop_ac", "_mdot_cp"):
            getattr(fresh, name).copy_(getattr(model, name))
        fresh._aux.zero_()
        if model.residual is not None:
            fresh.residual.load_state_dict(model.residual.state_dict())
    return fresh
