"""``etm fit-plant`` and ``etm eval-plant``.

Identification runs on day-grouped folds and reports open-loop rollout error
against both baselines.  The gate for L1 is stated in the roadmap: a 20-minute
rollout on held-out trips, with physically plausible parameters.  This command
either shows that or shows it failing -- it does not average the failure away.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from ..splits import COLD_HOLDOUT_TRIPS, cold_holdout, day_groups, grouped_folds
from .evaluate import HORIZONS_S, evaluate_plant, per_trip_rmse
from .fit import FitConfig, estimate_runtime, fit_plant, transfer_aux
from .windows import build_windows, load_trips

log = logging.getLogger("etm.plant")


def _load(args) -> tuple[dict, pd.DataFrame]:
    processed = Path(args.processed)
    trips = load_trips(processed, pattern=args.pattern)
    ov_path = processed / "overview.parquet"
    overview = pd.read_parquet(ov_path) if ov_path.exists() else None
    return trips, overview


def cmd_fit_plant(args) -> int:
    trips, overview = _load(args)
    names = sorted(trips)
    dev_trips, cold = cold_holdout(names)
    print(f"trips: {len(names)}  development: {len(dev_trips)}  "
          f"cold hold-out: {cold or 'none present'}")

    windows = build_windows({k: trips[k] for k in dev_trips},
                            horizon_s=args.horizon, stride_s=args.stride)
    print(f"windows: {len(windows)} x {windows.horizon}s from {len(windows.trips)} trips")

    epochs = args.epochs[0] if len(args.epochs) == 1 else tuple(args.epochs)
    if not isinstance(epochs, int) and len(epochs) != len(args.curriculum):
        raise SystemExit(f"--epochs takes 1 value or one per curriculum stage "
                         f"({len(args.curriculum)}); got {len(epochs)}")
    cfg = FitConfig(horizons_s=tuple(args.curriculum), epochs_per_stage=epochs,
                    lr=args.lr, batch_size=args.batch, device=args.device,
                    progress=not args.quiet)

    if overview is not None and args.folds > 1:
        groups = day_groups(overview)
        folds = grouped_folds(list(windows.trips), groups, n_splits=args.folds)
    else:
        folds = []

    n_runs = max(1, len(folds))
    eta = estimate_runtime(windows, cfg, n_folds=n_runs, use_residual=args.residual)
    print(f"projected runtime: {eta / 60:.0f} min for {n_runs} fold(s) on {cfg.device}"
          f"   ({eta / 60 / n_runs:.0f} min each)")
    if args.estimate_only:
        print("stopping here (--estimate-only)")
        return 0

    rows = []
    if folds:
        for spec in folds:
            tr = windows.select_trips(set(spec.train_trips))
            te = windows.select_trips(set(spec.test_trips))
            if len(tr) == 0 or len(te) == 0:
                continue
            res = fit_plant(tr, cfg=cfg, use_residual=args.residual)
            # A held-out trip has no fitted auxiliary term -- zero it, which is
            # the setting the model would face at deployment.
            model = transfer_aux(res.model, n_trips=len(te.trips))
            scores = evaluate_plant(model, te)
            p = res.model.params()
            print(f"\n[{spec.name}] test={', '.join(spec.test_trips)}")
            print(f"  {p}")
            if p.implausible():
                print(f"  IMPLAUSIBLE: {p.implausible()}")
            moved = res.model.movement_from_initial()
            stuck = {k: v for k, v in moved.items() if v < 0.05}
            if stuck:
                print("  BARELY MOVED from initialisation (undertrained or unexcited): "
                      + ", ".join(f"{k} {v:+.1%}" for k, v in sorted(stuck.items())))
            print(scores.table())
            rows.append({"fold": spec.name, **{f"rmse_{h}s": v for h, v in scores.plant.items()},
                         **{f"skill_{h}s": scores.skill(h) for h in scores.plant},
                         **p.__dict__})
    else:
        res = fit_plant(windows, cfg=cfg, use_residual=args.residual)
        p = res.model.params()
        print(f"\n{p}")
        if p.implausible():
            print(f"IMPLAUSIBLE: {p.implausible()}")
        print(evaluate_plant(res.model, windows).table())
        rows.append({"fold": "all", **p.__dict__})
        torch.save(res.model.state_dict(), Path(args.out) / "plant.pt")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "plant_folds.csv", index=False)

    if len(df) > 1:
        print("\n" + "=" * 60)
        print("ACROSS FOLDS (mean +/- sd)")
        print("=" * 60)
        for h in HORIZONS_S:
            col = f"rmse_{h}s"
            if col in df:
                print(f"  {h:>5}s rollout RMSE : {df[col].mean():.3f} +/- {df[col].std():.3f} C"
                      f"   skill {df[f'skill_{h}s'].mean():+.1%}")
        print("\n  identified parameters:")
        for k in ("c_cab", "c_mass", "ua0", "ua1", "ua_mass", "eta", "tau_h"):
            if k in df:
                print(f"    {k:<9} {df[k].mean():>12,.2f} +/- {df[k].std():>10,.2f}")
        print("\n  Cross-fold spread alone does not prove identification: parameters")
        print("  that never left their initialisation also agree perfectly. Read the")
        print("  spread together with the movement warnings above.")
    return 0


def cmd_eval_plant(args) -> int:
    trips, _ = _load(args)
    windows = build_windows(trips, horizon_s=args.horizon, stride_s=args.stride)
    from .model import RCPlant

    model = RCPlant(n_trips=len(windows.trips), use_residual=args.residual)
    state = torch.load(args.model, map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("_aux")}
    model.load_state_dict(state, strict=False)

    print(model.params())
    print(evaluate_plant(model, windows).table())
    print("\nworst trips at 300 s:")
    for name, rmse in list(per_trip_rmse(model, windows, 300).items())[:8]:
        print(f"    {name:<10} {rmse:6.3f} C")
    return 0


def register(sub) -> None:
    common = dict()
    pf = sub.add_parser("fit-plant", help="identify the grey-box cabin thermal model")
    pf.add_argument("--processed", default="data/processed")
    pf.add_argument("--out", default="artifacts/plant")
    pf.add_argument("--pattern", default="TripB*.parquet",
                    help="winter trips by default; the summer schemas lack the coolant circuit")
    pf.add_argument("--horizon", type=int, default=1200, help="window length in seconds")
    pf.add_argument("--stride", type=int, default=300)
    pf.add_argument("--curriculum", type=int, nargs="+", default=[120, 300, 600, 1200])
    pf.add_argument("--epochs", type=int, nargs="+", default=[60],
                    help="one value for every stage, or one per curriculum stage "
                         "(e.g. --epochs 400 300 150 80 to taper as the horizon grows)")
    pf.add_argument("--lr", type=float, default=3e-2)
    pf.add_argument("--batch", type=int, default=64,
                    help="a rollout is 1200 sequential RK4 steps on tiny tensors, so "
                         "the GPU is launch-bound; use a batch as large as memory allows "
                         "(--batch 96 covers the whole window set) or the CPU may well win")
    pf.add_argument("--folds", type=int, default=5)
    pf.add_argument("--residual", action="store_true", help="add the bounded neural correction")
    pf.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pf.add_argument("--estimate-only", action="store_true",
                    help="print the projected runtime and exit without fitting")
    pf.add_argument("--quiet", action="store_true", help="suppress the progress line")
    pf.set_defaults(func=cmd_fit_plant)

    pe = sub.add_parser("eval-plant", help="score a saved plant against baselines")
    pe.add_argument("--model", required=True)
    pe.add_argument("--processed", default="data/processed")
    pe.add_argument("--pattern", default="TripB*.parquet")
    pe.add_argument("--horizon", type=int, default=1200)
    pe.add_argument("--stride", type=int, default=600)
    pe.add_argument("--residual", action="store_true")
    pe.set_defaults(func=cmd_eval_plant)
