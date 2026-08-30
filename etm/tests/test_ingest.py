import numpy as np
import pandas as pd
import pytest

from etm.ingest import HEATER_RATING_W, TripQuality, _canonicalise, resample
from etm.schema import MEASUREMENT_MAP


def test_duplicate_columns_are_merged_not_dropped():
    """The raw headers carry 'Velocity [km/h]' twice-over in TripB38, each half empty."""
    raw = pd.DataFrame({
        "Velocity [km/h]": [1.0, np.nan, 3.0],
        "Velocity [km/h]]]": [np.nan, 2.0, np.nan],
    })
    q = TripQuality(trip="t")
    out = _canonicalise(raw, MEASUREMENT_MAP, q)
    assert list(out.columns) == ["velocity_kmh"]
    assert out["velocity_kmh"].tolist() == [1.0, 2.0, 3.0]
    assert q.merged_duplicate_columns == ["velocity_kmh"]
    assert not q.conflicting_duplicates


def test_conflicting_duplicates_are_kept_separately_not_merged():
    """'Temperature Vent right' appears twice per file but holds two real sensors."""
    raw = pd.DataFrame({
        "Temperature Vent right [°C]": [10.0, 11.0],
        "Temperature Vent right [°C] ": [10.0, 99.0],
    })
    q = TripQuality(trip="t")
    out = _canonicalise(raw, MEASUREMENT_MAP, q)
    assert q.conflicting_duplicates == ["vent_right_c"]
    assert list(out.columns) == ["vent_right_c", "vent_right_c__2"]
    assert out["vent_right_c"].tolist() == [10.0, 11.0]
    assert out["vent_right_c__2"].tolist() == [10.0, 99.0]


def test_unmapped_columns_are_recorded():
    q = TripQuality(trip="t")
    _canonicalise(pd.DataFrame({"Totally Unknown [x]": [1]}), MEASUREMENT_MAP, q)
    assert q.unmapped_columns == ["Totally Unknown [x]"]


def test_resample_block_means_to_1hz():
    df = pd.DataFrame({
        "trip": ["t"] * 20,
        "time_s": np.arange(20) * 0.1,
        "heat_power_req_w": np.arange(20, dtype=float),
    })
    out = resample(df, period_s=1.0)
    assert len(out) == 2
    assert out["heat_power_req_w"].tolist() == [4.5, 14.5]
    assert out["time_s"].tolist() == [0.0, 1.0]
    assert out["trip"].tolist() == ["t", "t"]


def test_resample_requires_time():
    with pytest.raises(KeyError):
        resample(pd.DataFrame({"trip": ["t"], "x": [1.0]}))


def test_heater_rating_is_the_documented_limit():
    assert HEATER_RATING_W == 7000.0


def test_trailing_empty_column_is_not_reported_as_a_lost_signal():
    """A trailing ';' on TripA11's header line makes pandas invent 'Unnamed: 23'."""
    raw = pd.DataFrame({
        "Time [s]": [0.0, 0.1],
        "Unnamed: 23": [np.nan, np.nan],
    })
    q = TripQuality(trip="t")
    out = _canonicalise(raw, MEASUREMENT_MAP, q)
    assert q.trailing_empty_columns == ["Unnamed: 23"]
    assert q.unmapped_columns == []
    assert list(out.columns) == ["time_s"]


def test_placeholder_header_with_real_data_is_still_flagged_unmapped():
    raw = pd.DataFrame({"Unnamed: 7": [1.0, 2.0]})
    q = TripQuality(trip="t")
    _canonicalise(raw, MEASUREMENT_MAP, q)
    assert q.unmapped_columns == ["Unnamed: 7"]
    assert q.trailing_empty_columns == []


def test_schema_fingerprint_separates_instrumentation_variants():
    """Winter trips carry 48 channels; summer trips come in reduced variants."""
    from etm.ingest import schema_fingerprint
    winter = ["time_s", "velocity_kmh", "cabin_temp_c", "coolant_inlet_c", "vent_right_c"]
    summer = ["time_s", "velocity_kmh", "cabin_temp_c"]
    assert schema_fingerprint(winter) != schema_fingerprint(summer)
    # order and duplication must not matter
    assert schema_fingerprint(winter) == schema_fingerprint(reversed(winter))
    assert schema_fingerprint(summer) == schema_fingerprint(summer + ["time_s"])


def test_sparse_placeholder_column_is_classified_as_a_corrupt_record():
    """TripA11 has exactly one line with an extra field, not a 14 245-row channel."""
    col = [np.nan] * 500
    col[4] = 43650.0
    q = TripQuality(trip="t")
    _canonicalise(pd.DataFrame({"Time [s]": np.arange(500) * 0.1,
                                "Unnamed: 23": col}), MEASUREMENT_MAP, q)
    assert q.stray_field_columns == ["Unnamed: 23(1)"]
    assert q.unmapped_columns == []
    assert q.trailing_empty_columns == []


def test_dense_placeholder_column_is_a_real_unmapped_signal():
    q = TripQuality(trip="t")
    _canonicalise(pd.DataFrame({"Unnamed: 9": np.arange(500, dtype=float)}), MEASUREMENT_MAP, q)
    assert q.unmapped_columns == ["Unnamed: 9"]
    assert q.stray_field_columns == []


def test_overview_blank_spacer_rows_are_dropped(tmp_path):
    """Overview.xlsx has two blank rows between the summer and winter blocks."""
    import pandas as pd
    from etm.ingest import load_overview
    path = tmp_path / "Overview.xlsx"
    pd.DataFrame({
        "Trip": ["TripA01", None, "TripB01", "  "],
        "Date": ["2019-06-25_13-21-14", None, "2019_12_07_12-35-00", None],
        "Target Cabin Temperature": [23.0, None, 22.0, None],
        "Note": ["", None, "directly after previous trip", None],
    }).to_excel(path, index=False)
    ov = load_overview(path)
    assert ov["trip"].tolist() == ["TripA01", "TripB01"]
    assert not ov["trip"].duplicated().any()
    assert ov["day"].tolist() == ["2019-06-25", "2019-12-07"]
    assert ov["warm_start"].tolist() == [False, True]
