"""Command line entry points.

    python -m etm ingest  --root "D:/Electro-thermal-modelling" --out data/processed
    python -m etm audit   --out data/processed

``ingest`` is the only step that touches the raw CSVs.  Everything downstream
reads the canonical parquet store, so the parsing quirks documented in
:mod:`etm.ingest` are dealt with exactly once.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import ingest as ing
from . import splits as sp

log = logging.getLogger("etm")


def _cmd_ingest(args: argparse.Namespace) -> int:
    root, out = Path(args.root), Path(args.out)
    meas_dir = root / "Measurement Data"
    sim_dir = root / "Simulation Data"
    out.mkdir(parents=True, exist_ok=True)

    quality = ing.ingest_measurements(meas_dir, out, period_s=args.period,
                                      pattern=args.pattern)
    print(f"ingested {len(quality)} trips -> {out/'trips'} @ {args.period}s")

    ov_path = meas_dir / "Overview.xlsx"
    if ov_path.exists():
        overview = ing.load_overview(ov_path)
        overview.to_parquet(out / "overview.parquet", index=False)
        print(f"overview: {len(overview)} trips, "
              f"{overview['day'].nunique()} distinct days, "
              f"setpoints (all trips)="
              f"{sorted(float(v) for v in overview['cabin_setpoint_c'].dropna().unique())} °C")

    if not args.skip_sim and sim_dir.exists():
        n = ing.ingest_simulations(sim_dir, out)
        print(f"ingested {n} simulation runs -> {out/'sim'}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Print the data-quality findings that must not be silently inherited."""
    out = Path(args.out)
    q = pd.read_csv(out / "quality_report.csv")

    print("=" * 72)
    print("DATA QUALITY AUDIT")
    print("=" * 72)
    print(f"trips              : {len(q)}")
    print(f"rows raw -> stored : {q.n_rows_raw.sum():,} -> {q.n_rows_out.sum():,}")
    periods = sorted({round(float(v), 4) for v in q.dt_median_s.dropna()})
    print(f"sample period      : {', '.join(f'{v:g}' for v in periods)} s")

    # --- instrumentation variants -----------------------------------------
    # The corpus is not one dataset.  Anything trained on one variant must
    # declare which signals the others lack.
    variants = q.groupby("schema_id", sort=False).agg(
        trips=("trip", list), lo=("n_raw_columns", "min"), hi=("n_raw_columns", "max"))
    if len(variants) > 1:
        print(f"\ninstrumentation variants: {len(variants)} distinct sensor sets")
        for sid, r in variants.sort_values("trips", key=lambda s: s.map(len),
                                           ascending=False).iterrows():
            trips = r.trips
            span = f"{int(r.lo)}" if r.lo == r.hi else f"{int(r.lo)}-{int(r.hi)}"
            head = ", ".join(trips[:4]) + (f", +{len(trips) - 4} more" if len(trips) > 4 else "")
            print(f"    {sid}  {span:>5} raw cols  {len(trips):>3} trips   {head}")
        print("    -> a model fitted on the richest variant cannot be applied to the others")
        print("       without declaring the missing signals")

    flat = q[q.target_constant.fillna(False).astype(bool)]
    if len(flat):
        print(f"\ntrips where the target never changes ({len(flat)}): heater idle for the whole trip")
        print(f"    {', '.join(flat.trip)}")
        print("    -> expected for summer trips; a winter trip here would be a fault")

    over = q[q.n_target_over_rating > 0]
    if len(over):
        print(f"\ntarget above 7 kW heater rating (dropped, not clipped): "
              f"{int(q.n_target_over_rating.sum())} rows in {len(over)} trips")
        for r in over.sort_values("target_max_raw_w", ascending=False).head(5).itertuples():
            print(f"    {r.trip:<9} {r.n_target_over_rating:>6} rows, max {r.target_max_raw_w:,.0f} W")

    merged = q[q.merged_duplicate_columns.fillna("") != ""]
    if len(merged):
        print(f"\nduplicate/variant headers merged in {len(merged)} trips:")
        for name, grp in merged.groupby("merged_duplicate_columns"):
            print(f"    {name:<28} {len(grp)} trips")

    conflict = q[q.conflicting_duplicates.fillna("") != ""]
    if len(conflict):
        print("\nsame-named headers holding DIFFERENT signals (kept separately as '__2'):")
        for name, grp in conflict.groupby("conflicting_duplicates"):
            print(f"    {name:<28} {len(grp)} trips")

    unmapped = q[q.unmapped_columns.fillna("") != ""]
    if len(unmapped):
        print(f"\nUNMAPPED raw columns -- real signals being dropped, add them to the schema:")
        for name, grp in unmapped.groupby("unmapped_columns"):
            print(f"    {name!r:<40} {len(grp)} trips")

    empty = q[q.trailing_empty_columns.fillna("") != ""]
    if len(empty):
        print(f"\nempty placeholder columns from a trailing ';' (ignored): "
              f"{len(empty)} trips")

    stray = q[q.stray_field_columns.fillna("") != ""]
    if len(stray):
        print("\ncorrupt records -- a few lines carry an extra field (value dropped):")
        for r in stray.itertuples():
            print(f"    {r.trip:<9} {r.stray_field_columns}   (count in parentheses)")

    const = q[q.constant_columns.fillna("") != ""]
    if len(const):
        from collections import Counter
        counts = Counter(c for s in const.constant_columns for c in str(s).split(","))
        print("\nconstant-within-trip channels (no information; do not engineer features from these):")
        for col, n in counts.most_common(10):
            print(f"    {col:<28} constant in {n}/{len(q)} trips")

    ov_path = out / "overview.parquet"
    if ov_path.exists():
        ov = pd.read_parquet(ov_path)
        b = ov[ov.trip.str.startswith("TripB")]
        print("\n" + "-" * 72)
        print("DESIGN-OF-EXPERIMENT COVERAGE (winter trips)")
        print("-" * 72)
        print(f"cabin setpoint  : {sorted(float(v) for v in b.cabin_setpoint_c.dropna().unique())} °C  "
              "<- single setpoint: no basis for generalising to others")
        print(f"ambient (start) : {b.amb_temp_start_c.min():.1f} .. "
              f"{b.amb_temp_start_c.max():.1f} °C")
        print(f"trips / days    : {len(b)} trips over {b.day.nunique()} days")
        warm = b[b.warm_start]
        print(f"warm-start trips: {len(warm)} ({', '.join(warm.trip)})")
        sess = sp.session_groups(b)
        print(f"driving sessions: {sess.nunique()}  <- the real independent-sample count")
        multi = b.groupby("day").trip.count()
        print(f"days with >1 trip: {int((multi > 1).sum())} of {len(multi)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="etm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="parse raw CSVs into the canonical parquet store")
    pi.add_argument("--root", required=True, help="folder holding 'Measurement Data' and 'Simulation Data'")
    pi.add_argument("--out", default="data/processed")
    pi.add_argument("--period", type=float, default=1.0, help="resample period in seconds")
    pi.add_argument("--pattern", default="Trip*.csv")
    pi.add_argument("--skip-sim", action="store_true")
    pi.set_defaults(func=_cmd_ingest)

    pa = sub.add_parser("audit", help="print the data-quality and coverage report")
    pa.add_argument("--out", default="data/processed")
    pa.set_defaults(func=_cmd_audit)

    try:
        from .plant.cli import register as register_plant
    except ImportError:  # torch not installed: the data layer still works
        pass
    else:
        register_plant(sub)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
