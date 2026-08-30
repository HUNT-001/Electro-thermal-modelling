"""Turn the canonical parquet store into rollout windows for plant identification.

A "window" is a contiguous slice of one trip: an initial cabin temperature, an
input sequence, and the measured cabin temperature to score against.  Windows
never straddle trips, and the trip index travels with each one so the model can
look up that trip's auxiliary heat term.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .model import INPUT_CHANNELS

__all__ = ["WindowSet", "build_windows", "load_trips"]

#: Canonical columns the plant model consumes, in :data:`INPUT_CHANNELS` order.
_SOURCE = {
    "p_heat_w": "heat_power_req_w",
    "p_ac_w": "aircon_power_w",
    "amb_temp_c": "amb_temp_c",
    "speed_ms": "velocity_kmh",     # converted below
}
TARGET_COL = "cabin_temp_c"


@dataclass
class WindowSet:
    """Batched rollout windows, ready for the model."""

    u: Tensor            # (N, T, 4) inputs
    y: Tensor            # (N, T)    measured cabin temperature
    trip_idx: Tensor     # (N,)      index into `trips`
    start_s: Tensor      # (N,)      window start time within its trip
    trips: list[str]

    def __len__(self) -> int:
        return self.u.shape[0]

    @property
    def horizon(self) -> int:
        return self.u.shape[1]

    def to(self, device: str | torch.device) -> "WindowSet":
        return WindowSet(self.u.to(device), self.y.to(device), self.trip_idx.to(device),
                         self.start_s, self.trips)

    def subset(self, mask: np.ndarray | Tensor) -> "WindowSet":
        m = torch.as_tensor(mask)
        return WindowSet(self.u[m], self.y[m], self.trip_idx[m], self.start_s[m], self.trips)

    def truncate(self, horizon: int) -> "WindowSet":
        """Shorten every window -- used by the horizon curriculum during fitting."""
        h = min(horizon, self.horizon)
        return WindowSet(self.u[:, :h], self.y[:, :h], self.trip_idx, self.start_s, self.trips)

    def select_trips(self, keep: set[str]) -> "WindowSet":
        idx = {i for i, t in enumerate(self.trips) if t in keep}
        mask = torch.tensor([int(i) in idx for i in self.trip_idx.tolist()])
        return self.subset(mask)


def load_trips(processed_dir: Path, pattern: str = "Trip*.parquet") -> dict[str, pd.DataFrame]:
    """Read the canonical per-trip parquet files, keeping only usable trips."""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(processed_dir).glob(f"trips/{pattern}")):
        df = pd.read_parquet(path)
        if TARGET_COL not in df.columns:
            continue
        out[path.stem] = df
    if not out:
        raise FileNotFoundError(f"no usable trips under {processed_dir}/trips")
    return out


def _inputs(df: pd.DataFrame) -> np.ndarray:
    """Assemble the (T, 4) input array, tolerating missing optional channels."""
    n = len(df)
    cols = []
    for name in INPUT_CHANNELS:
        src = _SOURCE[name]
        if src in df.columns:
            v = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=np.float64)
        else:
            # Summer trips lack no input; but the reduced schemas can lack A/C.
            # Absent actuation is zero actuation -- an honest default, and the
            # quality report already records which trips are missing what.
            v = np.zeros(n)
        if name == "speed_ms":
            v = v / 3.6
        cols.append(v)
    u = np.stack(cols, axis=-1)
    return u


def build_windows(
    trips: dict[str, pd.DataFrame],
    horizon_s: int = 1200,
    stride_s: int = 300,
    dt_s: float = 1.0,
    max_gap_frac: float = 0.02,
) -> WindowSet:
    """Slice every trip into overlapping windows of ``horizon_s`` seconds.

    Windows whose inputs or target are more than ``max_gap_frac`` missing are
    dropped rather than imputed: a rollout is a simulation, and filling a gap in
    the heater trace with a median invents energy that never entered the cabin.
    Short remaining gaps are linearly interpolated.
    """
    steps = int(round(horizon_s / dt_s))
    stride = max(1, int(round(stride_s / dt_s)))

    names = sorted(trips)
    u_list, y_list, idx_list, start_list = [], [], [], []

    for ti, name in enumerate(names):
        df = trips[name].reset_index(drop=True)
        u_full = _inputs(df)
        y_full = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
        t_full = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=np.float64)
        if len(df) < steps:
            continue

        for s in range(0, len(df) - steps + 1, stride):
            e = s + steps
            u_w, y_w = u_full[s:e], y_full[s:e]
            miss = np.isnan(u_w).any(axis=-1) | np.isnan(y_w)
            if miss.mean() > max_gap_frac or miss[0]:
                continue
            if miss.any():
                good = ~miss
                grid = np.arange(steps)
                y_w = np.interp(grid, grid[good], y_w[good])
                u_w = np.stack([np.interp(grid, grid[good], u_w[good, c])
                                for c in range(u_w.shape[1])], axis=-1)
            u_list.append(u_w)
            y_list.append(y_w)
            idx_list.append(ti)
            start_list.append(t_full[s])

    if not u_list:
        raise ValueError(
            f"no windows of {horizon_s}s survived; longest trip is "
            f"{max(len(d) for d in trips.values())} samples")

    return WindowSet(
        u=torch.tensor(np.stack(u_list), dtype=torch.float32),
        y=torch.tensor(np.stack(y_list), dtype=torch.float32),
        trip_idx=torch.tensor(idx_list, dtype=torch.long),
        start_s=torch.tensor(start_list, dtype=torch.float32),
        trips=names,
    )
