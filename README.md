# Electro-Thermal Modelling — from a heating-power predictor to a cabin-heating controller

Predicting, forecasting, and **controlling** the cabin-heating power of a battery EV
in winter, on the [BMW i3 real-driving dataset][dataset]. The goal is the one an
energy-management system actually cares about: reach the same cabin comfort the
production controller achieves, using less heater energy — and therefore more
winter range.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-grey?logo=pytorch)
![Tests](https://img.shields.io/badge/tests-~110%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> **Status.** The `etm/` package (L0–L3 below) is complete and reproducible.
> An earlier version of this repo trained a two-stage XGBoost model to predict
> instantaneous heating power; that work, its corrected results, and why the
> project was rebuilt are described under *[What changed, and why](#what-changed-and-why)*.

---

## The result in one line

On held-out winter trips, a model-predictive controller acting on a thermal model
**identified from the data** reaches the production controller's comfort using
**23.7 % ± 6.3 % less heater energy** — about **+0.6 km of winter range per 20 minutes**
of driving — and keeps that advantage under ±20 % error in the identified plant.

This is a **simulation** result, with the limits stated plainly [below](#honest-limits).
It is built on four layers, each with baselines it has to beat and each reporting
its own failures rather than only its wins.

---

## The four layers

| Layer | What it does | Headline | Beats |
|---|---|---|---|
| **L0 · Data** | Canonical schema + ingest that fixes every defect in the raw CSVs; day/session-grouped CV; cold hold-out | 70 trips, 4 instrumentation variants, all quirks handled once | — |
| **L1 · Plant** | Differentiable grey-box RC cabin thermal model, identified by multi-step rollout loss | 20-min warm-up predicted to **~2.8 °C RMSE**, **+54–70 % skill** | persistence, exponential-decay |
| **L2 · Forecast** | Probabilistic multi-horizon demand forecast with calibrated intervals | **+24–51 %** skill at 60–900 s; interval coverage 0.74–0.83 after conformal | persistence, warm-up curve |
| **L3 · Control** | MPC by gradient descent through the L1 plant | **+23.7 %** energy saved at OEM-equivalent comfort | OEM policy, thermostat, PI |

Each layer is a package under `etm/` with its own CLI subcommand and test suite.
Full technical detail, per-fold numbers, and the reasoning behind every design
choice live in **[`etm/README.md`](etm/README.md)**.

---

## Quickstart

```bash
cd etm
pip install -e ".[dev,model,control]"

# 1. parse the raw CSVs into a canonical parquet store (fixes the data defects)
python -m etm ingest --root ".." --out data/processed
python -m etm audit  --out data/processed          # what the data will and won't support

# 2. identify the cabin thermal plant (L1)
python -m etm fit-plant --epochs 200 150 100 60 --folds 5

# 3. forecast heating demand with calibrated intervals (L2)
python -m etm forecast --folds 5

# 4. run MPC vs the OEM / thermostat / PI baselines, with a robustness sweep (L3)
python -m etm control --n-trips 8 --robustness

pytest        # ~110 tests
```

The raw measurement and simulation CSVs are not redistributed here (size, and the
[dataset's own terms][dataset]); point `--root` at your local copy.

---

## What changed, and why

The original model predicted instantaneous heating power, `P(t)`, from the vehicle's
sensors at time `t`, using a two-stage XGBoost classifier + regressor. Re-examining
it turned up four things worth correcting, and they are the reason for the rebuild:

- **The headline was a single split.** Re-run as 6-fold cross-validation, that model
  scores **R² = 0.83 ± 0.11**, not the reported 0.736 (an unlucky fold) or 0.833
  (a lucky one). The spread is the result.
- **The idle-detection stage doesn't work.** "Heater off" exists in essentially
  **two** of the 38 winter trips — both fast-charging stops — so the classifier's
  cross-validated F1 is 0.45–0.51, not the reported 0.99. It was learning two events.
- **A predictor is not a controller.** `P(t)` from sensors at `t` is a *soft sensor*:
  by the time you know the answer, the energy is spent. Scheduling battery power
  needs a *forecast* (L2) and a *plant model* (L1) to optimise a *controller* against
  (L3). That is the whole point of the rebuild.
- **The dataset is narrower than it looks.** One vehicle, one city, a single 22 °C
  setpoint, ambient only −3…+9 °C, and **35 driving sessions** — not 627 k independent
  samples, because 10 Hz rows an hour apart are not independent. Every result here is
  reported against that reality.

The rebuild keeps the good parts of the original (the dataset choice, the physical
feature intuition) and rebuilds the rest as a tested, honest, layered package.

---

## Honest limits

Stated up front, because the credibility of the rest depends on it:

- **L3 is a simulation study.** A recorded trip cannot be re-driven with a different
  heater policy, so the controllers act on the L1 plant. Internal validity is good —
  every controller faces the *same* plant, so model error largely cancels in the
  comparison, and the ±20 % robustness sweep shows the ranking is stable — but the
  plant's ~2.9 °C cabin RMSE is large next to a 1 °C comfort band, so external validity
  is the open question. The one measured bias runs *against* the controller (simulation
  flatters the OEM), so 23.7 % is if anything conservative.
- **Speed-dependent heat loss (`UA₁`) is not yet identified** — only three highway trips
  carry the excitation. Motorway-trip savings are the least trustworthy until this is fixed.
- **The OEM controller optimises more than cabin air temperature** — demisting, humidity,
  vent-level comfort — which this cost function does not model. "Beating" it on cabin
  temperature alone is not beating it at its full job.
- **Single setpoint, mild winter.** Nothing here supports a claim about other cabin
  setpoints or the sub-zero temperatures where EV range loss actually bites.

---

## Roadmap

- [x] **L0** Data platform, quality audit, grouped CV
- [x] **L1** Grey-box thermal plant, identified and gated
- [x] **L2** Probabilistic demand forecaster with conformal calibration
- [x] **L3** MPC vs OEM/thermostat/PI, with a robustness sweep
- [ ] **Fix `UA₁`** so motorway heat loss — and every claim that depends on it — is valid
- [ ] **Safety supervisor** — hard actuator limits, out-of-distribution flag, deterministic
  fallback to the OEM controller: the piece an OEM would need to evaluate this
- [ ] **RL comparison arm** (SAC / offline RL) against the same environment and baselines
- [ ] **Production surface** — config, tracking, ONNX export, embedded-latency budget

---

## Repository layout

```
etm/                     the rebuilt package — install and run from here
  etm/schema.py, ingest.py, splits.py     L0 data platform
  etm/plant/                              L1 grey-box thermal model
  etm/forecast/                           L2 probabilistic demand forecaster
  etm/control/                            L3 MPC and the controller comparison
  tests/                                  ~110 tests, each pinned to a real behaviour
  README.md                               full technical detail and per-fold results
scripts/                 the original two-stage XGBoost pipeline (kept for reference)
Measurement Data/        raw BMW i3 trips (not redistributed)
Simulation Data/         companion physics-simulation runs (−10 °C, unused by L1–L3 so far)
```

## Dataset & licence

Measurements and simulation runs: *Battery and Heating Data in Real Driving Cycles*,
BMW i3 (60 Ah), M. Steinstraeter et al., TU Munich, published on
[IEEE DataPort][dataset] — see the dataset page for its own terms.
Code in this repository is released under the MIT licence.

[dataset]: https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles
