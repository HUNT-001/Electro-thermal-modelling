"""Canonical signal schema for the BMW i3 electro-thermal dataset.

The raw CSVs use inconsistent, locale-mangled headers (cp1252 degree signs, BOMs,
stray brackets, one duplicated column, and one file with a malformed velocity
header).  Everything downstream speaks the canonical names defined here, so the
mess is confined to this module.

Design rules
------------
* Normalisation is aggressive and lossy on purpose: it must map
  ``Velocity [km/h]`` and the malformed ``Velocity [km/h]]]`` (TripB38) onto the
  same canonical signal so the two half-empty columns can be merged.
* Every canonical name carries an explicit unit.  Unit conversions live in
  :data:`SCALE`, applied once at ingest.
* Signals are tagged by *causal tier* so feature ablations are declarative
  rather than regex-guesswork scattered through training scripts.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

__all__ = [
    "Tier",
    "normalise",
    "MEASUREMENT_MAP",
    "SIMULATION_MAP",
    "SCALE",
    "OFFSET",
    "TIER",
    "TARGET",
    "columns_in_tiers",
]


class Tier(str, Enum):
    """Causal tier of a signal relative to the heater.

    Used to build honest ablations.  A model that only sees ``UPSTREAM`` and
    ``VEHICLE`` signals is making a genuine prediction; one that sees
    ``HEATER_CIRCUIT`` is largely reading the heater's own thermal footprint,
    and one that sees ``ACTUATOR`` is reading the answer.
    """

    INDEX = "index"           # time, trip id
    UPSTREAM = "upstream"     # ambient conditions, route -- causes, not effects
    VEHICLE = "vehicle"       # dynamics and battery -- independent of the heater
    CABIN = "cabin"           # cabin air temperature -- effect, but a legitimate sensor
    HEATER_CIRCUIT = "heater_circuit"  # coolant / vent / defrost air -- direct heater footprint
    ACTUATOR = "actuator"     # heater voltage/current/signal, delivered power -- leakage
    TARGET = "target"


#: The variable being modelled: heater power the HVAC controller *requests*.
#: Note this is not the same as delivered power (``heat_power_can_w`` /
#: ``heat_power_lin_w``), which diverges from it by >350 W mean absolute on
#: TripB04, TripB09 and TripB34.
TARGET = "heat_power_req_w"

_DEGREE_RE = re.compile(r"[°º�]")
_BRACKET_RE = re.compile(r"[\[\]\(\)\{\}]")
_SEP_RE = re.compile(r"[/\\\-_.,;:+]")
_WS_RE = re.compile(r"\s+")


def normalise(raw: str) -> str:
    """Reduce a raw CSV header to a stable lookup key.

    Strips BOMs, degree signs, all bracket characters (which is what makes the
    malformed ``Velocity [km/h]]]`` collapse onto ``velocity km h``), separator
    punctuation, and case.

    >>> normalise("Velocity [km/h]") == normalise("Velocity [km/h]]]")
    True
    >>> normalise("\\ufeffBattery Temperature [\\u00b0C]")
    'battery temperature c'
    """
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace("﻿", " ")
    s = _DEGREE_RE.sub(" ", s)
    s = _BRACKET_RE.sub(" ", s)
    s = _SEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def _m(*raw_headers: str) -> tuple[str, ...]:
    return tuple(normalise(h) for h in raw_headers)


# --------------------------------------------------------------------------
# Measurement data (Measurement Data/Trip*.csv)
# --------------------------------------------------------------------------

#: normalised raw header -> canonical name
MEASUREMENT_MAP: dict[str, str] = {}


def _reg(canonical: str, *raw_headers: str) -> None:
    for key in _m(*raw_headers):
        if key in MEASUREMENT_MAP and MEASUREMENT_MAP[key] != canonical:
            raise ValueError(f"header key {key!r} already mapped to {MEASUREMENT_MAP[key]!r}")
        MEASUREMENT_MAP[key] = canonical


_reg("time_s", "Time [s]")
# Route / environment -- causes
_reg("amb_temp_c", "Ambient Temperature [°C]")
_reg("amb_temp_sensor_c", "Ambient Temperature Sensor [°C]")
_reg("elevation_m", "Elevation [m]")
# Vehicle dynamics
_reg("velocity_kmh", "Velocity [km/h]")           # also absorbs "Velocity [km/h]]]"
_reg("throttle_pct", "Throttle [%]")
_reg("motor_torque_nm", "Motor Torque [Nm]")
_reg("accel_long_ms2", "Longitudinal Acceleration [m/s^2]")
_reg("regen_signal", "Regenerative Braking Signal")
# Battery
_reg("batt_voltage_v", "Battery Voltage [V]")
_reg("batt_current_a", "Battery Current [A]")
_reg("batt_temp_c", "Battery Temperature [°C]")
_reg("batt_temp_max_c", "max. Battery Temperature [°C]")
_reg("soc_pct", "SoC [%]")
_reg("soc_displayed_pct", "displayed SoC [%]")
_reg("soc_min_pct", "min. SoC [%]")
_reg("soc_max_pct", "max. SoC [%)")               # sic: mismatched bracket in the raw file
# Cabin
_reg("cabin_temp_c", "Cabin Temperature Sensor [°C]")
# Heater coolant circuit
_reg("coolant_heatercore_c", "Coolant Temperature Heatercore [°C]")
_reg("coolant_req_c", "Requested Coolant Temperature [°C]")
_reg("coolant_inlet_c", "Coolant Temperature Inlet [°C]")
_reg("coolant_flow_lph", "Coolant Volume Flow +500 [l/h]")
_reg("hx_temp_c", "Heat Exchanger Temperature [°C]")
_reg("hx_out_c", "Temperature Heat Exchanger Outlet [°C]")
_reg("heater_coolant_in_c", "Temperature Coolant Heater Inlet [°C]")
_reg("heater_coolant_out_c", "Temperature Coolant Heater Outlet [°C]")
# Vent / duct air temperatures
_reg("vent_defrost_lat_l_c", "Temperature Defrost lateral left [°C]")
_reg("vent_defrost_lat_r_c", "Temperature Defrost lateral right [°C]")
_reg("vent_defrost_ctr_c", "Temperature Defrost central [°C]")
_reg("vent_defrost_ctr_l_c", "Temperature Defrost central left [°C]")
_reg("vent_defrost_ctr_r_c", "Temperature Defrost central right [°C]")
_reg("vent_footwell_drv_c", "Temperature Footweel Driver [°C]")
_reg("vent_footwell_pax_c", "Temperature Footweel Co-Driver [°C]")
_reg("vent_feet_drv_c", "Temperature Feetvent Driver [°C]")
_reg("vent_feet_pax_c", "Temperature Feetvent Co-Driver [°C]")
_reg("vent_head_drv_c", "Temperature Head Driver [°C]")
_reg("vent_head_pax_c", "Temperature Head Co-Driver [°C]")
_reg("vent_right_c", "Temperature Vent right [°C]")   # duplicated in the raw header
_reg("vent_ctr_r_c", "Temperature Vent central right [°C]")
_reg("vent_ctr_l_c", "Temperature Vent central left [°C]")
# Actuator / target
_reg("heat_power_req_w", "Requested Heating Power [W]")
_reg("heat_power_can_w", "Heating Power CAN [kW]")
_reg("heat_power_lin_w", "Heating Power LIN [W]")
_reg("aircon_power_w", "AirCon Power [kW]")
_reg("heater_signal", "Heater Signal")
_reg("heater_voltage_v", "Heater Voltage [V]")
_reg("heater_current_a", "Heater Current [A]")


#: Multiplicative unit fixes applied once, at ingest.
SCALE: dict[str, float] = {
    "heat_power_can_w": 1000.0,   # kW -> W
    "aircon_power_w": 1000.0,     # kW -> W
}

#: Additive offsets baked into the raw signal names.
#: The coolant flow channel is logged with a +500 bias (see the raw header
#: "Coolant Volume Flow +500 [l/h]"), so the physical flow is value - 500.
OFFSET: dict[str, float] = {
    "coolant_flow_lph": -500.0,
}


TIER: dict[str, Tier] = {
    "time_s": Tier.INDEX,
    # upstream
    "amb_temp_c": Tier.UPSTREAM,
    "amb_temp_sensor_c": Tier.UPSTREAM,
    "elevation_m": Tier.UPSTREAM,
    # vehicle
    "velocity_kmh": Tier.VEHICLE,
    "throttle_pct": Tier.VEHICLE,
    "motor_torque_nm": Tier.VEHICLE,
    "accel_long_ms2": Tier.VEHICLE,
    "regen_signal": Tier.VEHICLE,
    "batt_voltage_v": Tier.VEHICLE,
    "batt_current_a": Tier.VEHICLE,
    "batt_temp_c": Tier.VEHICLE,
    "batt_temp_max_c": Tier.VEHICLE,
    "soc_pct": Tier.VEHICLE,
    "soc_displayed_pct": Tier.VEHICLE,
    "soc_min_pct": Tier.VEHICLE,
    "soc_max_pct": Tier.VEHICLE,
    # cabin
    "cabin_temp_c": Tier.CABIN,
    # heater circuit
    "coolant_heatercore_c": Tier.HEATER_CIRCUIT,
    "coolant_req_c": Tier.HEATER_CIRCUIT,
    "coolant_inlet_c": Tier.HEATER_CIRCUIT,
    "coolant_flow_lph": Tier.HEATER_CIRCUIT,
    "hx_temp_c": Tier.HEATER_CIRCUIT,
    "hx_out_c": Tier.HEATER_CIRCUIT,
    "heater_coolant_in_c": Tier.HEATER_CIRCUIT,
    "heater_coolant_out_c": Tier.HEATER_CIRCUIT,
    **{
        c: Tier.HEATER_CIRCUIT
        for c in (
            "vent_defrost_lat_l_c", "vent_defrost_lat_r_c", "vent_defrost_ctr_c",
            "vent_defrost_ctr_l_c", "vent_defrost_ctr_r_c", "vent_footwell_drv_c",
            "vent_footwell_pax_c", "vent_feet_drv_c", "vent_feet_pax_c",
            "vent_head_drv_c", "vent_head_pax_c", "vent_right_c",
            "vent_ctr_r_c", "vent_ctr_l_c",
        )
    },
    # actuator (leakage)
    "heat_power_can_w": Tier.ACTUATOR,
    "heat_power_lin_w": Tier.ACTUATOR,
    "aircon_power_w": Tier.ACTUATOR,
    "heater_signal": Tier.ACTUATOR,
    "heater_voltage_v": Tier.ACTUATOR,
    "heater_current_a": Tier.ACTUATOR,
    # target
    "heat_power_req_w": Tier.TARGET,
}


def columns_in_tiers(*tiers: Tier) -> list[str]:
    """Canonical column names belonging to any of ``tiers``, in schema order."""
    wanted = set(tiers)
    return [c for c, t in TIER.items() if t in wanted]


# --------------------------------------------------------------------------
# Simulation data (Simulation Data/*.csv)
# --------------------------------------------------------------------------
# These files carry a three-row header: machine names, column-group names, then
# human-readable names with units.  Several human-readable names repeat
# ("Power [kW]" appears under both Battery and Consumption), so the simulation
# loader resolves columns positionally against the group row rather than by
# name alone.

SIMULATION_MAP: dict[str, str] = {
    normalise("Time [s]"): "time_s",
    normalise("Ambient Temperature [C]"): "amb_temp_c",
    normalise("Slope [rad]"): "slope_rad",
    normalise("Wind Speed [m/s]"): "wind_speed_ms",
    normalise("Longitudinal Acceleration [m/s^2]"): "accel_long_ms2",
    normalise("Speed [m/s]"): "speed_ms",
    normalise("Position [m]"): "position_m",
    normalise("Drive Power [W]"): "drive_power_w",
    normalise("Pack Voltage [V]"): "batt_voltage_v",
    normalise("Battery Current [A]"): "batt_current_a",
    normalise("SOC [%]"): "soc_pct",
    normalise("Mean Cell Temperature [C]"): "batt_temp_c",
    normalise("Cabin Temperature [C]"): "cabin_temp_c",
    normalise("Heating Power [kW]"): "heat_power_req_w",
    normalise("Heating Power inclPeak [kW]"): "heat_power_peak_w",
    normalise("Heatexchanger Inlet Coolant Temp [C]"): "coolant_inlet_c",
    normalise("Heatexchanger Outlet Air Temp [C]"): "hx_out_c",
    normalise("Blower [W]"): "blower_w",
    normalise("Auxiliaries Total [W]"): "aux_total_w",
}

#: Column group each simulation signal belongs to, used to disambiguate the
#: repeated human-readable names ("Power [kW]").
SIMULATION_GROUPS = ("Environment", "Longitudinal Vehicle", "Drive Train",
                     "Battery", "Consumption", "High Voltage Heater", "Auxiliaries")

SIM_SCALE: dict[str, float] = {
    "heat_power_req_w": 1000.0,
    "heat_power_peak_w": 1000.0,
}
