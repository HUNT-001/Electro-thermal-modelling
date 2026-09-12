"""``etm control`` -- run the controller comparison and the robustness sweep.

Produces the number the project stands on: heater energy saved at the comfort
the production controller actually achieved, per trip, plus what that means in
kilometres of winter range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..splits import cold_holdout
from .cost import ComfortSpec
from .env import ThermalEnv, TripScenario, make_plant
from .evaluate import evaluate_trip, range_gain_km, robustness_sweep, run_controller

DEFAULT_WEIGHTS = (2.0, 3.0, 5.0, 8.0, 15.0)


def _consumption_wh(df, max_s: float) -> float:
    """Total battery energy over the scored span, from measured V and I."""
    if not {"batt_voltage_v", "batt_current_a"}.issubset(df.columns):
        return float("nan")
    n = min(len(df), int(max_s))
    v = pd.to_numeric(df["batt_voltage_v"], errors="coerce").to_numpy()[:n]
    i = pd.to_numeric(df["batt_current_a"], errors="coerce").to_numpy()[:n]
    p = np.nan_to_num(v * i)
    return float(np.abs(p).sum() / 3600.0)


def _require_trips(processed: Path, pattern: str = "TripB*.parquet") -> list[Path]:
    """Fail immediately, and say where we looked, when the store is not there.

    A run that quietly finds zero trips and then dies in ``pd.concat`` eighty
    lines later tells the user nothing.  The usual cause is a relative
    ``--processed`` resolved against the wrong working directory.
    """
    trips_dir = processed / "trips"
    if not trips_dir.is_dir():
        raise SystemExit(
            f"no trip store at {trips_dir.resolve()}\n"
            f"  --processed is relative to the current directory ({Path.cwd()}).\n"
            f"  Run `python -m etm ingest --root <data root> --out {processed}` first,\n"
            f"  or point --processed at an existing store.")
    paths = sorted(trips_dir.glob(pattern))
    if not paths:
        raise SystemExit(
            f"{trips_dir.resolve()} contains no files matching {pattern!r}\n"
            f"  (found {len(list(trips_dir.glob('*.parquet')))} parquet files in total)")
    return paths


def cmd_control(args) -> int:
    processed = Path(args.processed)
    paths = _require_trips(processed)
    plant = make_plant()
    if args.plant and Path(args.plant).exists():
        import torch
        plant = make_plant(state_dict=torch.load(args.plant, map_location="cpu"))
    print(f"plant: {plant.params()}")

    overview = None
    ov = processed / "overview.parquet"
    if ov.exists():
        overview = pd.read_parquet(ov).set_index("trip")

    names = [p.stem for p in paths]
    dev, cold = cold_holdout(names)
    chosen = dev[:args.n_trips] if args.n_trips else dev
    spec = ComfortSpec(band_c=args.band)

    print(f"controllers on {len(chosen)} trips, control interval {args.dt}s, "
          f"first {args.max_s}s of each trip\n")

    detail, summaries, skipped = [], [], []
    for name in chosen:
        df = pd.read_parquet(processed / "trips" / f"{name}.parquet")
        if len(df) < args.max_s:
            skipped.append(name)
            continue
        sc = TripScenario.from_trip(name, df, dt_s=args.dt, max_s=args.max_s)
        rows, summary = evaluate_trip(plant, sc, spec, weights=DEFAULT_WEIGHTS,
                                      iters=args.iters, replan_every=args.replan_every)
        if overview is not None and name in overview.index:
            dist = float(overview.loc[name, "distance_km"])
            frac = float(overview.loc[name, "duration_min"]) * 60.0
            dist_scored = dist * min(1.0, args.max_s / max(frac, 1.0))
            saved = summary["oem_energy_wh"] - summary["mpc_energy_at_oem_comfort_wh"]
            summary["range_gain_km"] = range_gain_km(
                saved, dist_scored, _consumption_wh(df, args.max_s))
        detail.append(rows.assign(trip=name))
        summaries.append(summary)
        s = summary
        flag = "" if s["reachable"] else "   (OEM comfort outside swept range)"
        print(f"  {name}: OEM {s['oem_energy_wh']:6.0f} Wh @ {s['oem_discomfort_kmin']:6.1f} K-min"
              f"  ->  MPC {s['mpc_energy_at_oem_comfort_wh']:6.0f} Wh"
              f"  saving {s['energy_saving_frac']:+6.1%}{flag}")

    if not detail:
        raise SystemExit(
            f"every candidate trip was shorter than --max-s ({args.max_s:g}s): "
            f"{', '.join(skipped) or 'none found'}\n"
            f"  lower --max-s, or pick trips that run longer than the scored span")
    if skipped:
        print(f"\n  skipped (shorter than {args.max_s:g}s): {', '.join(skipped)}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sm = pd.DataFrame(summaries)
    sm.to_csv(out / "control_summary.csv", index=False)
    pd.concat(detail).to_csv(out / "control_detail.csv", index=False)
    _report(sm)

    if args.robustness:
        print("\n" + "=" * 72)
        print("ROBUSTNESS -- MPC plans with the nominal plant, is scored on a wrong one")
        print("=" * 72)
        df = pd.read_parquet(processed / "trips" / f"{chosen[0]}.parquet")
        sc = TripScenario.from_trip(chosen[0], df, dt_s=args.dt, max_s=args.max_s)
        rb = robustness_sweep(sc, spec, comfort_weight=5.0,
                              iters=args.iters, replan_every=args.replan_every)
        rb.to_csv(out / "control_robustness.csv", index=False)
        for r in rb.itertuples():
            d = r.energy_wh - r.oem_energy_wh
            print(f"  {r.perturbation:<14} MPC {r.energy_wh:6.0f} Wh @ {r.discomfort_kmin:6.1f} K-min"
                  f"   (OEM {r.oem_energy_wh:.0f} Wh @ {r.oem_discomfort_kmin:.1f})   "
                  f"delta {d:+6.0f} Wh")
        print("\n  A saving that only exists at the nominal parameters is a modelling")
        print("  artefact, not a saving. Read the spread, not the best row.")

    if args.supervised:
        _demo_supervisor(processed, chosen[0], spec, args)
    return 0


def _demo_supervisor(processed, trip_name, spec, args):
    """Show the deterministic safety layer: transparent in nominal operation,
    and containing three injected faults with an auditable takeover log."""
    from .controllers import MPCController, PIController
    from .supervisor import SupervisedController
    print("\n" + "=" * 72)
    print("SAFETY SUPERVISOR -- deterministic layer with final say over the heater")
    print("=" * 72)
    df = pd.read_parquet(processed / "trips" / f"{trip_name}.parquet")
    sc = TripScenario.from_trip(trip_name, df, dt_s=args.dt, max_s=args.max_s)
    env = ThermalEnv(make_plant(), sc, spec)

    def fresh_mpc():
        return MPCController(comfort_weight=5.0, spec=spec, iters=args.iters,
                             replan_every=args.replan_every)

    sup = SupervisedController(primary=fresh_mpc(), fallback=PIController())
    r = run_controller(env, sup, "supervised")
    print(f"  nominal: supervised MPC {r.energy_wh:.0f} Wh @ {r.discomfort_kmin:.1f} K-min, "
          f"peak {r.peak_w:.0f} W")
    print("  " + sup.report.summary().replace("\n", "\n  "))

    # fault injection: a primary that periodically emits NaN and over-range commands
    class Faulty:
        def __init__(self, inner): self.inner = inner
        def reset(self): self.inner.reset()
        def __call__(self, k, t_cab, env):
            if k % 17 == 0:
                return float("nan")
            if k % 11 == 0:
                return 40000.0
            return self.inner(k, t_cab, env)
    sup2 = SupervisedController(primary=Faulty(fresh_mpc()), fallback=PIController())
    _, p = env.rollout(sup2)
    print(f"\n  fault-injected primary (periodic NaN + 40 kW commands):")
    print(f"    heater stayed within [0, {p.max():.0f}] W, all finite = {bool(np.isfinite(p).all())}")
    print("    " + sup2.report.summary().replace("\n", "\n    "))
    print("\n  The learned controller proposes; this layer disposes. Every override is")
    print("  logged -- what an OEM certifying the system would require.")


def _report(sm: pd.DataFrame) -> None:
    ok = sm[sm.reachable & sm.energy_saving_frac.notna()]
    print("\n" + "=" * 72)
    print("ENERGY SAVED AT OEM-EQUIVALENT COMFORT")
    print("=" * 72)
    if not len(ok):
        print("  no trip reached the OEM comfort level within the swept weights --")
        print("  widen DEFAULT_WEIGHTS rather than reporting a partial result")
        return
    print(f"  trips scored           : {len(ok)} of {len(sm)}")
    print(f"  energy saving          : {ok.energy_saving_frac.mean():+.1%} "
          f"+/- {ok.energy_saving_frac.std():.1%}")
    print(f"  OEM heater energy      : {ok.oem_energy_wh.mean():.0f} Wh mean over the scored span")
    if "range_gain_km" in ok and ok.range_gain_km.notna().any():
        g = ok.range_gain_km.dropna()
        print(f"  equivalent winter range: {g.mean():+.2f} km +/- {g.std():.2f} "
              f"over the scored span")
    print("\n  This is a SIMULATION result. The controllers act on the L1 grey-box plant,")
    print("  because TripB05 cannot be re-driven with a different heater policy. Its")
    print("  credibility rests on that plant, which reproduces a 20-minute warm-up to")
    print("  ~2.8 degC with +70% skill over naive baselines -- and on the robustness sweep.")


def register(sub) -> None:
    pc = sub.add_parser("control", help="MPC vs OEM/thermostat/PI on the learned plant")
    pc.add_argument("--processed", default="data/processed")
    pc.add_argument("--out", default="artifacts/control")
    pc.add_argument("--plant", default="", help="path to a fitted plant.pt (optional)")
    pc.add_argument("--dt", type=float, default=10.0, help="control interval, seconds")
    pc.add_argument("--max-s", type=float, default=1200.0, help="scored span per trip")
    pc.add_argument("--band", type=float, default=1.0, help="comfort deadband, K")
    pc.add_argument("--n-trips", type=int, default=0, help="0 = all development trips")
    pc.add_argument("--iters", type=int, default=25)
    pc.add_argument("--replan-every", type=int, default=12)
    pc.add_argument("--robustness", action="store_true")
    pc.add_argument("--supervised", action="store_true",
                    help="demonstrate the safety supervisor: nominal transparency + fault containment")
    pc.set_defaults(func=cmd_control)
