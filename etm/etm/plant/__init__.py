"""Grey-box cabin thermal plant: identification, rollout evaluation, baselines."""

from .model import PlantParams, RCPlant
from .windows import WindowSet, build_windows, load_trips
from .fit import FitConfig, FitResult, fit_plant, rollout_rmse, transfer_aux
from .evaluate import BaselineScores, evaluate_plant, per_trip_rmse, energy_balance_check

__all__ = [
    "PlantParams", "RCPlant", "WindowSet", "build_windows", "load_trips",
    "FitConfig", "FitResult", "fit_plant", "rollout_rmse", "transfer_aux",
    "BaselineScores", "evaluate_plant", "per_trip_rmse", "energy_balance_check",
]
