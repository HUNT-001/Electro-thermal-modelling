# etm — electro-thermal demand forecasting and cabin-heating control for BEVs

Phase 0 foundation: a canonical data layer and an evaluation design that the
rest of the project can be trusted to stand on.

## Why this exists

The original pipeline reported a single train/validation/test split. Re-run as
6-fold cross-validation, the same model scores **R² = 0.83 ± 0.11** — the
published 0.736 was an unlucky fold and 0.833 a lucky one. Before adding
sequence models, a learned plant and a controller, the evaluation has to be
capable of telling improvement from fold noise. That is what Phase 0 builds.

## Install

```bash
pip install -e ".[dev]"          # data layer + tests
pip install -e ".[model,control,track]"   # later phases
```

## Use

```bash
python -m etm ingest --root "D:/Electro-thermal-modelling" --out data/processed
python -m etm audit  --out data/processed
pytest
```

`ingest` is the only code that touches the raw CSVs. Everything downstream
reads `data/processed/`, so the parsing quirks are handled exactly once.

## What the data layer fixes

| Problem in the raw data | What happens now |
|---|---|
| **TripB38** header carries a malformed extra `Velocity [km/h]]]`; a naive loader leaves velocity empty for all 16 429 rows of that trip and median-imputes it | Bracket-insensitive normalisation maps both columns to `velocity_kmh` and merges them. Verified: 0 % missing, mean 41.4 km/h |
| `Temperature Vent right` appears **twice** in every file, holding two different duct sensors | Kept separately as `vent_right_c` and `vent_right_c__2`. Merged only where duplicates actually agree |
| 1 599 rows exceed the 7 kW heater rating, including a **38 528 W** spike in TripB02 | Dropped and counted in `quality_report.csv`, not silently clipped to 7 000 |
| `Requested Coolant Temperature` is **constant at 85 °C** in every winter row, so `85 − cabin_temp` is just negated cabin temperature | Flagged as a constant channel in the audit so it is not re-engineered into a feature |
| `Coolant Volume Flow +500 [l/h]` carries a +500 bias in its own name | Offset removed at ingest |
| Simulation files have a **three-row header** with repeated names | Resolved positionally by `load_simulation_run` |
| `Overview.xlsx` (route, weather, payload, fan, setpoint) was never loaded | Joined as trip metadata; drives the day/session grouping |
| **The 70 trips are not one dataset** — winter trips carry 48 channels, summer trips come in three reduced instrumentation levels (23, 28, and 22 for TripA21) with no coolant circuit and no vent or defrost temperatures | Every trip carries a `schema_id` fingerprint; `audit` lists the variants so nothing is trained across incompatible sensor sets by accident |
| **TripA11** has one corrupt line carrying an extra field (`43650` at t = 0.4 s) | Classified as a corrupt record and reported, distinct from a genuinely unmapped signal |
| 10 Hz sampling makes 627 k rows look like 627 k independent samples | Resampled to 1 Hz by block mean; `effective_sample_size` estimates the true independent count for confidence intervals |

## Evaluation design

- **Group by day, not trip.** Five of the six original test trips had a
  same-day, same-route sibling in train.
- **Sessions keep warm-start chains intact.** TripB16/B17 and B28 are logged
  "directly after previous trip" — they inherit cabin and coolant state from
  their predecessor, which is the most informative variable for a warm-up model.
- **Cold hold-out.** TripB37 and TripB38 are the only sub-zero trips; they are
  reserved so cold-weather claims are extrapolation tests, not interpolation.
- **Honest confidence intervals.** `effective_sample_size` uses the
  initial-positive-sequence estimator so serial correlation is not mistaken for
  evidence.

## Coverage limits to state in any write-up

- One vehicle (BMW i3, 60 Ah), one city (Munich), one driver population.
- Cabin setpoint is **22 °C in all 38 winter trips** — nothing supports a claim
  about other setpoints.
- Ambient spans **−3 °C to +9 °C**. Real winter range loss happens well below
  that; the simulation set (−10 °C) is the only probe of it.
- Idle (heater off) exists in essentially **two trips**, both fast-charging
  stops. Any idle-detection metric on this data is measuring those two events.

## Layout

```
etm/
├── schema.py    canonical names, units, causal tiers
├── ingest.py    raw CSV → canonical parquet, all quirks handled here
├── splits.py    day/session grouping, cold hold-out, effective sample size
└── cli.py       `etm ingest`, `etm audit`
tests/           21 tests, each pinned to a real defect in the raw data
```

## L1 — grey-box cabin thermal plant

`etm/plant/` identifies a 3-state RC model (cabin air, interior mass,
delivered-heat lag) by minimising **free-running rollout error**, never
one-step error. Run it with:

```bash
python -m etm fit-plant --folds 5 --epochs 400 --batch 96 --device cuda
```

**Check the cost before committing.** A rollout is 1200 sequential RK4 steps
over a `(batch, 3)` tensor, so the run is launch-bound and a GPU is often no
faster than the CPU here. `--estimate-only` times one step per curriculum stage
and projects the whole run:

```bash
python -m etm fit-plant --epochs 400 --folds 5 --batch 96 --estimate-only
# projected runtime: 153 min for 5 fold(s) on cpu   (31 min each)
```

Cost is linear in horizon x epochs, so the 1200 s stage dominates. Taper the
epochs as the horizon grows and most of that goes away for very little loss —
the long stage is refining slow dynamics, not discovering them:

```bash
python -m etm fit-plant --epochs 400 300 150 80 --folds 5 --batch 96
# projected runtime: 64 min for 5 fold(s)
```

Fitting prints a live progress line with elapsed time and ETA; `--quiet`
suppresses it.

Identification is verified by `test_identification_recovers_known_parameters`:
trajectories are generated from known parameters and the fit has to recover
them. That test is the difference between "the optimiser reduced a loss" and
"the model identified the system". Both fitting tests are marked `slow`
(~15 min on CPU, far less on GPU) and excluded from the default run; use
`pytest -m slow` for them.

### First result on the 36 development winter trips — the L1 gate is not passed yet

| Horizon | Plant RMSE | Skill vs best baseline |
|---|---|---|
| 60 s | 0.580 ± 0.073 °C | **−25.1 %** |
| 300 s | 2.053 ± 0.396 °C | **−8.8 %** |

The identified parameters are physically plausible and land close to the
first-principles estimates (C_cab ≈ 111 kJ/K, UA₀ ≈ 50 W/K, τ_h ≈ 37 s,
η ≈ 0.85). But the model **loses to persistence**, and the cross-fold spread —
UA₀ = 49.84 ± 0.80, C_cab = 111 284 ± 87 — is far too tight to be real
agreement on 40 epochs of training. `movement_from_initial()` exists because of
this run: parameters that never left their initialisation report perfect
cross-fold stability while having learned nothing, and that reads as success
unless something checks for it.

Diagnosis for the next pass, in order of suspicion:

1. **Undertrained.** 40 epochs × 2 batches is ~80 optimiser steps per curriculum
   stage; η did not move at all. Run hundreds on the GPU.
2. **Windows are dominated by the steady holding phase**, where persistence is
   unbeatable by construction. Stratify evaluation into warm-up and steady
   regimes rather than averaging them.
3. **`Q_aux` is fitted per trip but zeroed at test time.** Honest, but it means
   the held-out rollout carries whatever bias that term was absorbing.
4. **The cabin sensor may not measure bulk cabin air** — sited in the vent path
   it would respond far faster than the modelled state.
