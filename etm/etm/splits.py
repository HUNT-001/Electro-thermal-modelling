"""Evaluation splits that respect how this data was actually collected.

The original pipeline used a single ``GroupShuffleSplit`` over 38 trips, giving
6 test trips.  Re-running it as 6-fold cross-validation showed test R² of
0.83 ± 0.11 -- the reported single-split result of 0.736 was an unlucky fold and
0.833 was a lucky one.  Neither is "the" number.

Two further problems the trip-level grouping does not solve:

1. **Same-day siblings.**  Five of the six original test trips have a
   same-day, same-route sibling in train (B05/B06, B28/B27, B31/B30,
   B34/B32-B33, and B16/B15 which the log explicitly marks "directly after
   previous trip").  The car, the driver, the traffic and the weather are
   shared; these are not independent samples.
2. **No cold extrapolation test.**  Ambient spans only -3 °C to +9 °C.  The two
   sub-zero trips (B37, B38) are the only probe of the regime an EV winter-range
   product actually cares about, so they are reserved as a held-out
   extrapolation set rather than averaged into a CV score.

This module provides day-grouped CV, a session-aware grouping that keeps
warm-start chains intact, and the cold hold-out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

__all__ = ["SplitSpec", "day_groups", "session_groups", "cold_holdout",
           "grouped_folds", "effective_sample_size", "valid_trips"]

#: Trips whose ambient temperature is below freezing.  Held out entirely so
#: that any claim about cold-weather performance is an extrapolation test and
#: not an interpolation one.
COLD_HOLDOUT_TRIPS = ("TripB37", "TripB38")


@dataclass(frozen=True)
class SplitSpec:
    """A single evaluation fold."""

    name: str
    train_trips: tuple[str, ...]
    test_trips: tuple[str, ...]

    def mask(self, trip: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        return trip.isin(self.train_trips).to_numpy(), trip.isin(self.test_trips).to_numpy()


#: A real trip identifier.  Overview.xlsx also carries blank spacer rows
#: between the summer and winter blocks, which arrive here as NaN or "nan"
#: and would otherwise become duplicate group keys.
TRIP_ID = re.compile(r"^Trip[AB]\d+$", re.IGNORECASE)


def valid_trips(overview: pd.DataFrame) -> pd.DataFrame:
    """Drop rows of the overview that do not name a real trip.

    Ingest filters these at source, but a parquet written by an earlier version
    can still contain them, and the failure they cause downstream ("duplicate
    trip ids: ['nan']") points at the grouping rather than at the spreadsheet.
    """
    trip = overview["trip"].astype(str).str.strip()
    return overview[trip.str.match(TRIP_ID, na=False)]


def day_groups(overview: pd.DataFrame) -> pd.Series:
    """Map trip -> calendar day.  The coarsest defensible grouping."""
    return valid_trips(overview).set_index("trip")["day"]


def session_groups(overview: pd.DataFrame) -> pd.Series:
    """Map trip -> driving *session*.

    A session is a maximal run of consecutive trips on the same day where each
    trip after the first is flagged ``warm_start`` ("directly after previous
    trip" in the log).  Those trips inherit cabin and coolant thermal state
    from their predecessor -- splitting them across train and test leaks the
    initial condition, which is the single most informative variable for a
    warm-up model.
    """
    ov = valid_trips(overview).sort_values("trip").reset_index(drop=True)
    session_id: list[str] = []
    current: str | None = None
    prev_day: str | None = None
    for row in ov.itertuples():
        if current is None or row.day != prev_day or not row.warm_start:
            current = f"{row.day}#{row.trip}"
        session_id.append(current)
        prev_day = row.day
    return pd.Series(session_id, index=ov["trip"], name="session")


def cold_holdout(trips: list[str]) -> tuple[list[str], list[str]]:
    """Split ``trips`` into (development, cold hold-out)."""
    cold = [t for t in trips if t in COLD_HOLDOUT_TRIPS]
    dev = [t for t in trips if t not in COLD_HOLDOUT_TRIPS]
    return dev, cold


def grouped_folds(
    trips: list[str], groups: pd.Series, n_splits: int = 6, exclude_cold: bool = True
) -> list[SplitSpec]:
    """Build ``n_splits`` group-disjoint folds over ``trips``.

    ``groups`` maps trip -> group label (use :func:`day_groups` or
    :func:`session_groups`).  Folds are balanced by group size, and no group is
    ever split across train and test.
    """
    dev, _ = cold_holdout(trips) if exclude_cold else (list(trips), [])
    dev = sorted(dev)
    if groups.index.has_duplicates:
        dupes = groups.index[groups.index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate trip ids in the group map: {dupes}")
    g = groups.reindex(dev)
    if g.isna().any():
        missing = g[g.isna()].index.tolist()
        raise KeyError(f"no group label for trips: {missing}")
    idx = np.arange(len(dev))
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(idx, groups=g.to_numpy())):
        folds.append(SplitSpec(
            name=f"fold{k}",
            train_trips=tuple(dev[i] for i in tr),
            test_trips=tuple(dev[i] for i in te),
        ))
    return folds


def effective_sample_size(x: np.ndarray, max_lag: int = 5000) -> float:
    """Estimate the number of *independent* samples in a serial correlated signal.

    Uses the initial-positive-sequence estimator: ``n_eff = n / (1 + 2*sum(rho_k))``
    truncated at the first non-positive autocorrelation.

    This exists to keep the project honest about confidence intervals.  A
    10 Hz trip of 10 000 rows spanning a heater transient with a multi-minute
    time constant carries on the order of tens of independent observations, not
    ten thousand.  Reporting a standard error as ``sd / sqrt(627092)`` would
    understate uncertainty by two orders of magnitude.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float(n)
    x = x - x.mean()
    var = float(x @ x)
    if var <= 0:
        return float(n)
    total = 0.0
    for lag in range(1, min(max_lag, n - 1)):
        rho = float(x[:-lag] @ x[lag:]) / var
        if rho <= 0:
            break
        total += rho
    return float(n / (1.0 + 2.0 * total))
