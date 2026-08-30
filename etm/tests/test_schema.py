import pytest

from etm.schema import Tier, columns_in_tiers, normalise, MEASUREMENT_MAP, TIER


def test_malformed_velocity_header_collapses_onto_canonical():
    """TripB38 ships an extra 'Velocity [km/h]]]' column; both must map to one signal."""
    assert normalise("Velocity [km/h]") == normalise("Velocity [km/h]]]")
    assert MEASUREMENT_MAP[normalise("Velocity [km/h]]]")] == "velocity_kmh"


def test_normalise_strips_bom_and_degree_variants():
    assert normalise("﻿Battery Temperature [°C]") == "battery temperature c"
    assert normalise("Battery Temperature [�C]") == "battery temperature c"


def test_mismatched_bracket_in_raw_header_is_mapped():
    # The raw file really does contain 'max. SoC [%)'.
    assert MEASUREMENT_MAP[normalise("max. SoC [%)")] == "soc_max_pct"


def test_every_mapped_column_has_a_tier():
    missing = {c for c in MEASUREMENT_MAP.values() if c not in TIER}
    assert not missing, f"untiered canonical columns: {missing}"


def test_tier_selection_excludes_leakage():
    honest = columns_in_tiers(Tier.UPSTREAM, Tier.VEHICLE)
    assert "heater_current_a" not in honest
    assert "heater_coolant_out_c" not in honest
    assert "amb_temp_c" in honest and "velocity_kmh" in honest


def test_target_is_request_not_delivered_power():
    from etm.schema import TARGET
    assert TARGET == "heat_power_req_w"
    assert TIER["heat_power_can_w"] is Tier.ACTUATOR
