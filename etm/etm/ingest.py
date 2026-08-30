"""Ingest raw BMW i3 measurement and simulation CSVs into a canonical store.

Everything known to be wrong with the raw files is handled here, loudly:

* **TripB38 velocity** -- that file's header contains a malformed extra column
  ``Velocity [km/h]]]``.  Because :func:`etm.schema.normalise` strips brackets,
  both columns land on ``velocity_kmh`` and are merged, recovering the 16 429
  rows of velocity that a naive loader silently drops.
* **Duplicated ``Temperature Vent right``** -- present twice in every
  measurement header.  Duplicates that agree are merged; disagreement is
  reported rather than hidden.
* **Out-of-range target** -- 1 599 rows (0.26 %) exceed the 7 kW heater rating,
  including a 38.5 kW spike in TripB02.  These are recorded as a quality flag
  and set to NaN, not silently clipped.
* **Constant channels** -- ``Requested Coolant Temperature`` is 85 °C in every
  row of every winter trip.  Flagged so nobody engineers a feature out of it
  believing it varies.
* **Simulation three-row header** -- machine names / group names / human names,
  with repeated human names, resolved positionally.

Ingest is deliberately dependency-light (pandas + numpy only) so it can run
inside a constrained environment before the ML stack is installed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    MEASUREMENT_MAP,
    OFFSET,
    SCALE,
    SIM_SCALE,
    SIMULATION_MAP,
    TARGET,
    normalise,
)

log = logging.getLogger(__name__)

HEATER_RATING_W = 7000.0
#: Encodings/separators seen across the corpus, tried in order.
_DIALECTS = ((";", "cp1252"), (";", "latin1"), (";", "utf-8-sig"),
             (",", "cp1252"), (",", "latin1"))


_LIST_FIELDS = ("merged_duplicate_columns", "conflicting_duplicates", "unmapped_columns",
                "trailing_empty_columns", "stray_field_columns", "missing_expected",
                "constant_columns")

#: Headers pandas invents for unnamed columns, produced by a trailing separator.
_PLACEHOLDER_HEADER = re.compile(r"^\s*(unnamed:\s*\d+)?\s*$", re.IGNORECASE)


@dataclass
class TripQuality:
    """Per-trip data-quality record, persisted alongside the parquet."""

    trip: str
    n_rows_raw: int = 0
    n_rows_out: int = 0
    n_raw_columns: int = 0
    schema_id: str = ""
    dt_median_s: float = float("nan")
    dt_irregular: bool = False
    merged_duplicate_columns: list[str] = field(default_factory=list)
    conflicting_duplicates: list[str] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)
    trailing_empty_columns: list[str] = field(default_factory=list)
    stray_field_columns: list[str] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    n_target_over_rating: int = 0
    target_max_raw_w: float = float("nan")
    target_constant: bool = False
    frac_missing_by_column: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        for k in _LIST_FIELDS:
            d[k] = ",".join(d[k])
        d.pop("frac_missing_by_column")
        return d


def schema_fingerprint(canonical_columns: Iterable[str]) -> str:
    """Stable short id for the set of canonical signals a file provides.

    The corpus is not one dataset.  The 38 winter trips carry 48 raw channels;
    the summer trips come in three reduced instrumentation levels (23, 28 and,
    for TripA21, 22 columns) with no coolant circuit and no vent or defrost air
    temperatures at all.  A model trained on one fingerprint cannot be applied
    to another without saying which signals it lost, so the fingerprint travels
    with every trip in the quality report.
    """
    digest = hashlib.sha1(";".join(sorted(set(canonical_columns))).encode()).hexdigest()
    return digest[:6]


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a measurement CSV, trying each known dialect."""
    last: Exception | None = None
    for sep, enc in _DIALECTS:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except Exception as exc:  # pragma: no cover - dialect probing
            last = exc
            continue
        if df.shape[1] > 1:
            return df
    raise ValueError(f"could not parse {path.name}") from last


def _canonicalise(df: pd.DataFrame, mapping: dict[str, str], q: TripQuality) -> pd.DataFrame:
    """Rename raw headers to canonical names, merging duplicate/variant columns.

    Two raw columns that normalise to the same canonical signal are combined
    with :meth:`pandas.Series.combine_first` **only when they agree** wherever
    both are observed -- that is the TripB38 velocity case, where the malformed
    header split one signal across two half-empty columns.

    When they disagree they are not the same signal despite the identical
    header, so the second is kept separately as ``{canon}__2`` and the clash is
    recorded.  This is the ``Temperature Vent right`` case: the header appears
    twice in every measurement file but carries two distinct duct sensors.
    Merging them would silently discard a real channel.
    """
    out: dict[str, pd.Series] = {}
    for raw in df.columns:
        key = normalise(raw)
        canon = mapping.get(key)
        if canon is None:
            col = pd.to_numeric(df[raw], errors="coerce")
            n_present = int(col.notna().sum())
            if not _PLACEHOLDER_HEADER.match(str(raw)):
                # A named header with no mapping is a real signal being dropped.
                q.unmapped_columns.append(str(raw))
            elif n_present == 0:
                # Every line ends with a trailing ';', so pandas invents an empty
                # 'Unnamed: N' column.  Cosmetic, not a lost signal.
                q.trailing_empty_columns.append(str(raw))
            elif n_present <= max(1, int(0.01 * len(col))):
                # A handful of lines carry an extra field: corrupt records, not a
                # channel.  TripA11 has exactly one (43650 at t = 0.4 s).
                q.stray_field_columns.append(f"{raw}({n_present})")
            else:
                q.unmapped_columns.append(str(raw))
            continue
        col = pd.to_numeric(df[raw], errors="coerce")
        if canon not in out:
            out[canon] = col
            continue
        prev = out[canon]
        both = prev.notna() & col.notna()
        if both.any() and not np.allclose(prev[both], col[both], equal_nan=True):
            q.conflicting_duplicates.append(canon)
            alt = f"{canon}__2"
            if alt not in out:
                out[alt] = col
        else:
            q.merged_duplicate_columns.append(canon)
            out[canon] = prev.combine_first(col)
    return pd.DataFrame(out)


def _apply_units(df: pd.DataFrame, scale: dict[str, float]) -> pd.DataFrame:
    for col, factor in scale.items():
        if col in df:
            df[col] = df[col] * factor
    for col, delta in OFFSET.items():
        if col in df:
            df[col] = df[col] + delta
    return df


def load_measurement_trip(path: Path) -> tuple[pd.DataFrame, TripQuality]:
    """Load one ``Trip*.csv`` into canonical form.

    Returns the frame (10 Hz, unresampled) and its quality record.  The target
    is left in physical units with out-of-range samples set to NaN; no
    thresholding or clipping is applied here, so downstream code is free to
    choose its own active/idle definition.
    """
    q = TripQuality(trip=path.stem)
    raw = _read_raw(path)
    q.n_rows_raw = len(raw)
    q.n_raw_columns = raw.shape[1]

    df = _canonicalise(raw, MEASUREMENT_MAP, q)
    df = _apply_units(df, SCALE)
    q.schema_id = schema_fingerprint(df.columns)

    if "time_s" in df:
        dt = np.diff(df["time_s"].to_numpy())
        if dt.size:
            q.dt_median_s = float(np.nanmedian(dt))
            q.dt_irregular = bool(np.nanstd(dt) > 1e-6)

    if TARGET in df:
        tgt = df[TARGET]
        q.target_max_raw_w = float(tgt.max())
        q.target_constant = bool(tgt.nunique(dropna=True) <= 1)
        over = tgt > HEATER_RATING_W
        q.n_target_over_rating = int(over.sum())
        df.loc[over, TARGET] = np.nan          # flagged, not silently clipped
        df = df.loc[df[TARGET].notna()].reset_index(drop=True)
    else:
        q.missing_expected.append(TARGET)

    q.constant_columns = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    q.frac_missing_by_column = {c: float(df[c].isna().mean()) for c in df.columns}
    q.n_rows_out = len(df)

    df.insert(0, "trip", path.stem)
    return df, q


def resample(df: pd.DataFrame, period_s: float = 1.0) -> pd.DataFrame:
    """Downsample a 10 Hz trip to ``period_s`` by block mean.

    The raw logs are 10 Hz, but cabin and coolant thermal time constants are
    minutes.  At 10 Hz, adjacent rows are near-duplicates: the 627 k row count
    is roughly 63 k independent samples at 1 Hz, and far fewer in terms of
    thermal information.  Row-level metrics computed at 10 Hz therefore report
    confidence intervals that are far too tight.  Resampling here makes the
    effective sample size explicit instead of implicit.
    """
    if "time_s" not in df:
        raise KeyError("resample requires a 'time_s' column")
    bucket = np.floor(df["time_s"].to_numpy() / period_s).astype(np.int64)
    num = df.select_dtypes(include=[np.number])
    out = num.groupby(bucket, sort=True).mean()
    out["time_s"] = out.index.to_numpy() * period_s
    out.insert(0, "trip", df["trip"].iloc[0])
    return out.reset_index(drop=True)


def load_overview(path: Path) -> pd.DataFrame:
    """Load ``Overview.xlsx`` -- the per-trip metadata the original pipeline ignored.

    Carries route, weather, distance, duration, added payload, fan setting and
    the cabin temperature setpoint.  The setpoint is 22 °C for all 38 winter
    trips, which is precisely why it matters: any claim about generalising to
    other setpoints is unsupported by this dataset.
    """
    raw = pd.read_excel(path)
    ren = {
        "Trip": "trip",
        "Date": "date",
        "Route/Area": "route",
        "Weather": "weather",
        "Battery Temperature (Start) [°C]": "batt_temp_start_c",
        "Battery Temperature (End)": "batt_temp_end_c",
        "Battery State of Charge (Start)": "soc_start",
        "Battery State of Charge (End)": "soc_end",
        "Ambient Temperature (Start) [°C]": "amb_temp_start_c",
        "Target Cabin Temperature": "cabin_setpoint_c",
        "Distance [km]": "distance_km",
        "Duration [min]": "duration_min",
        "Fan": "fan",
        "Note": "note",
    }
    df = raw[[c for c in ren if c in raw.columns]].rename(columns=ren)
    df["trip"] = df["trip"].astype(str).str.strip()
    # The sheet carries two blank spacer rows between the summer and winter
    # blocks.  Left in, they become duplicate empty trip ids and break any
    # grouping that indexes by trip.
    df = df[df["trip"].str.fullmatch(r"Trip[AB]\d+", case=False, na=False)]
    df = df.drop_duplicates(subset="trip", keep="first").reset_index(drop=True)
    # Dates are inconsistently delimited: 2019-12-07_13-21-14 and 2019_12_08_09-38-26
    stamp = df["date"].astype(str).str.replace("_", "-", regex=False)
    df["day"] = stamp.str.extract(r"^(\d{4}-\d{2}-\d{2})")[0]
    df["payload_kg"] = (
        df["note"].astype(str)
        .str.extract(r"\+\s*(\d+)\s*kg", flags=re.IGNORECASE)[0]
        .astype(float).fillna(0.0)
    )
    # "directly after previous trip" marks a warm-start trip whose thermal state
    # is inherited -- these must not be split across train/test independently.
    df["warm_start"] = df["note"].astype(str).str.contains(
        "directly after previous", case=False, na=False)
    return df


def load_simulation_run(path: Path) -> pd.DataFrame:
    """Load one ``Simulation Data/*.csv``.

    These carry a three-row header (machine names, group names, human names)
    and repeated human-readable names, so columns are resolved positionally:
    row index 1 of the raw file holds the human names used for mapping.
    """
    raw = pd.read_csv(path, sep=";", skiprows=1, encoding="cp1252", low_memory=False)
    names = [str(x).strip() for x in raw.iloc[0]]
    body = raw.iloc[1:].reset_index(drop=True)

    out: dict[str, pd.Series] = {}
    for pos, name in enumerate(names):
        canon = SIMULATION_MAP.get(normalise(name))
        if canon is None or canon in out:
            continue  # first occurrence wins; repeats are other groups' "Power [kW]"
        out[canon] = pd.to_numeric(body.iloc[:, pos], errors="coerce")

    df = pd.DataFrame(out)
    for col, factor in SIM_SCALE.items():
        if col in df:
            df[col] = df[col] * factor
    if "speed_ms" in df:
        df["velocity_kmh"] = df["speed_ms"] * 3.6

    stem = path.stem                      # e.g. Neg10_Charging_CAN
    amb, strategy, bus = stem.split("_")
    df.insert(0, "run", stem)
    df["amb_setpoint_c"] = {"Neg10": -10.0, "0": 0.0, "Pos10": 10.0}[amb]
    df["strategy"] = strategy
    df["bus"] = bus
    return df


def ingest_measurements(
    data_dir: Path, out_dir: Path, period_s: float = 1.0, pattern: str = "Trip*.csv"
) -> pd.DataFrame:
    """Ingest every measurement trip to ``out_dir`` and return the quality table."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    (out_dir / "trips").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(data_dir.glob(pattern)):
        df, q = load_measurement_trip(path)
        res = resample(df, period_s=period_s)
        res.to_parquet(out_dir / "trips" / f"{path.stem}.parquet", index=False)
        rows.append(q.as_row())
        log.info("%s: %d raw -> %d @ %.3gs (%d over rating)",
                 path.stem, q.n_rows_raw, len(res), period_s, q.n_target_over_rating)
    quality = pd.DataFrame(rows)
    quality.to_csv(out_dir / "quality_report.csv", index=False)
    return quality


def ingest_simulations(sim_dir: Path, out_dir: Path) -> int:
    """Ingest every simulation run to ``out_dir/sim``.  Returns the run count."""
    sim_dir, out_dir = Path(sim_dir), Path(out_dir)
    (out_dir / "sim").mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(sim_dir.glob("*.csv")):
        load_simulation_run(path).to_parquet(out_dir / "sim" / f"{path.stem}.parquet",
                                             index=False)
        n += 1
    return n
