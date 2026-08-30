import numpy as np
import pandas as pd
import pytest

from etm.splits import (COLD_HOLDOUT_TRIPS, cold_holdout, day_groups,
                        effective_sample_size, grouped_folds, session_groups)


@pytest.fixture
def overview():
    return pd.DataFrame({
        "trip": ["TripB15", "TripB16", "TripB17", "TripB18", "TripB37", "TripB38"],
        "day": ["2020-01-07"] * 3 + ["2020-01-08", "2020-01-21", "2020-02-07"],
        "warm_start": [False, True, True, False, False, False],
    })


def test_warm_start_chain_stays_in_one_session(overview):
    s = session_groups(overview)
    # B15 -> B16 -> B17 were driven back to back; they share thermal state.
    assert s["TripB15"] == s["TripB16"] == s["TripB17"]
    assert s["TripB18"] != s["TripB15"]


def test_cold_trips_are_held_out(overview):
    dev, cold = cold_holdout(overview.trip.tolist())
    assert cold == list(COLD_HOLDOUT_TRIPS)
    assert not set(dev) & set(cold)


def test_folds_never_split_a_group(overview):
    groups = day_groups(overview)
    folds = grouped_folds(overview.trip.tolist(), groups, n_splits=2)
    for f in folds:
        assert not set(f.train_trips) & set(f.test_trips)
        train_days = {groups[t] for t in f.train_trips}
        test_days = {groups[t] for t in f.test_trips}
        assert not train_days & test_days
        assert not set(f.test_trips) & set(COLD_HOLDOUT_TRIPS)


def test_effective_sample_size_penalises_serial_correlation():
    rng = np.random.default_rng(0)
    white = rng.normal(size=20_000)
    # A smooth signal like a cabin temperature trace at 10 Hz.
    smooth = np.convolve(rng.normal(size=20_000 + 999), np.ones(1000) / 1000, "valid")
    assert effective_sample_size(white) > 0.5 * white.size
    assert effective_sample_size(smooth) < 0.05 * smooth.size


def test_duplicate_trip_ids_raise_a_clear_error():
    groups = pd.Series(["d1", "d2"], index=["TripB01", "TripB01"])
    with pytest.raises(ValueError, match="duplicate trip ids"):
        grouped_folds(["TripB01"], groups, n_splits=2)


def test_group_maps_ignore_blank_overview_rows():
    """A parquet written before the ingest fix still carries the spacer rows."""
    from etm.splits import valid_trips
    stale = pd.DataFrame({
        "trip": ["TripB01", "nan", "TripB02", None, "  "],
        "day": ["2019-12-07", None, "2019-12-07", None, None],
        "warm_start": [False, False, True, False, False],
    })
    assert valid_trips(stale)["trip"].tolist() == ["TripB01", "TripB02"]
    g = day_groups(stale)
    assert not g.index.has_duplicates
    assert g.to_dict() == {"TripB01": "2019-12-07", "TripB02": "2019-12-07"}
    assert session_groups(stale).index.tolist() == ["TripB01", "TripB02"]


def test_grouped_folds_survives_a_stale_overview():
    stale = pd.DataFrame({
        "trip": ["TripB01", "nan", "TripB02", "nan", "TripB03", "TripB04"],
        "day": ["d1", None, "d2", None, "d3", "d4"],
        "warm_start": [False] * 6,
    })
    folds = grouped_folds(["TripB01", "TripB02", "TripB03", "TripB04"],
                          day_groups(stale), n_splits=2)
    assert len(folds) == 2
