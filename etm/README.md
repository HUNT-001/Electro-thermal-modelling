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

### Where L1 stands

Skill against the better of two baselines, 5 folds, 660 epochs:

| Horizon | Skill |
|---|---|
| 60 s | −24.2 % |
| 300 s | +2.2 % |
| 1200 s | **+54.1 %** |

**The model wins decisively at the horizon a controller plans over and loses at
60 s.** That split is not a training problem — it survived a 10× increase in
epochs — and chasing it turned up the real constraint.

#### The interior state is not observable from a mid-trip window

Three states, only one measured. `Q_del` has a ~45 s time constant so its
initial error washes out. `T_mass` has a ~14–25 min one, and its error does not
decay — the interior keeps exchanging heat with the cabin, so an error at t=0
compounds. Scored from the first second, the model is being judged on a guess.

Three fixes were tried and measured, and the first two failed:

| Approach | `Q_del` | `T_mass` |
|---|---|---|
| Guess `T_mass = T_cab` | wrong at first | wrong, and grows |
| Simulate through a burn-in | settles correctly | **worse** — 300 s RMSE 0.60 → 1.95 °C |
| Observer over measured history | recovers to 4 figures | still 12 K out |

No initialisation trick fixes an unobservable state. What does is starting
where the state is known: at key-on, after an overnight soak, the interior and
the cabin air are both at ambient. `--anchor trip_start` (now the default)
takes one window per trip from t=0 instead of sliding across it — 20 windows
instead of 78, but every one of them starts from a state that is actually
known. A vehicle is in the same position: its estimator runs from key-on and
never cold-starts mid-journey.

First fold under that configuration: 1200 s skill **+66.8 %**, 60 s −12.3 %
(from −24.2 %), C_mass 41 kJ/K and τ_mass 13.7 min, both plausible.

#### Parameters that cannot be identified are now frozen, not fitted

The A/C compressor runs in 0–6 % of winter rows and averages a few watts, so
`cop_ac` has almost no gradient — and a 660-epoch run duly drifted it to
6.0–7.3, above any physical COP, while the loss barely noticed. `unidentifiable()`
detects an unexcited input, holds the parameter at its initial value, and the
fit reports `NOT IDENTIFIED` rather than printing a number with no information
in it. The same check covers `UA₁` when speed does not vary.

#### One bound was moved, on the record

`C_mass`'s floor went from 50 kJ/K to 20 kJ/K. The original was a guess at the
whole interior's heat capacity; five folds independently agreed on
34.2 ± 2.2 kJ/K, and what couples to cabin air inside twenty minutes is the
surface layer — seat and trim skin, glass — not the full mass. Five folds beat
one guess. The bound moves once, not every time the data disagrees.

### Still open

- **UA₁ collapses to ~0** — the speed dependence of envelope loss is not
  identified even with speed varying. Only three highway trips (B10, B12, B14)
  carry real excitation and they are split across folds. Until this is fixed
  the model will understate heat loss at motorway speed, which is exactly the
  case long-trip range prediction depends on.
- **The 60 s deficit is reduced but not gone.** Worth deciding whether it
  matters: an MPC planning over 5–20 min does not need 60 s accuracy, and the
  honest move may be to state the model's valid horizon rather than chase it.
- **`--anchor trip_start` costs 3/4 of the windows** (20 of 78), because only
  trips longer than the horizon qualify. Multiple shooting — fitting each
  window's initial `T_mass` as a free parameter — would recover them.
- **`Q_aux` is fitted per trip but zeroed at test time.** Honest, but the
  held-out rollout carries whatever bias that term was absorbing.

## L2 — probabilistic demand forecaster

`etm/forecast/` predicts heating-power *demand* over several horizons from
information available now, so an EMS can reserve battery power **before** the
heater draws it. This is the opposite of the original repo's model, which
predicted `P_heat(t)` from sensors at `t` — a soft sensor, useless for
scheduling because the energy is already spent by the time you have the answer.

```bash
python -m etm forecast --processed data/processed --folds 5
```

### What is and isn't a feature

Legitimate at time `t`: the current and past heater power (autoregression on
the controller's own output is fair — an EMS has that signal live), the cabin
thermal state (cabin deficit below the 22 °C setpoint is the real driver of
future demand), ambient, battery, SoC, and the elapsed-time warm-up clock. Plus
future covariates a vehicle genuinely knows ahead of time — elapsed time at the
horizon, and on a real car the route elevation and nav speed. **Never** a
sensor's value at `t+h`: the heater-circuit temperatures the old model leaned on
are consequences of the heater running, and using their future value to predict
future power is the leakage this project exists to avoid.

### Result on the 36 development winter trips (5 day-grouped folds)

| Horizon | Forecaster pinball | vs best baseline |
|---|---|---|
| 30 s | 89.9 ± 31.1 W | +5.8 % |
| 60 s | 93.8 ± 30.4 W | +24.4 % |
| 300 s | 121.1 ± 53.4 W | +40.5 % |
| 600 s | 135.1 ± 58.8 W | +41.5 % |
| 900 s | 124.5 ± 48.6 W | +51.1 % |

Quantile gradient boosting (one model per horizon × quantile), against two
baselines it has to beat: persistence (`P(t+h) = P(t)`) and a warm-up curve
(median demand as a function of elapsed time). Beating the warm-up curve is what
proves the model uses the *thermal state*, not just the clock. The gain grows
with horizon — exactly where persistence fails and pre-scheduling pays off. At
30 s the +5.8 % edge is within the fold spread; short-horizon demand is
essentially "what it is now", and that's honest to state.

### The intervals have to mean what they say — and at first they didn't

An EMS sizes its reserve against the p90, so a miscalibrated interval is worse
than no interval. The raw quantile models were **systematically overconfident**:
the p10–p90 interval covered 0.50–0.68 of outcomes instead of 0.80, across every
fold. The cause is between-trip variation — quantiles calibrated within the
training trips never saw the spread a new winter trip brings.

Conformalized quantile regression fixes it with a finite-sample guarantee.
Within each fold, a slice of *training trips* is held out purely to measure how
far outside its own interval the model actually lands, and the interval is
inflated by that empirical miss. Because the calibration units are whole trips,
the correction accounts for exactly the between-trip variation the raw quantiles
missed:

| Horizon | Raw coverage | Conformal coverage | Conformal width |
|---|---|---|---|
| 30 s | 0.68 | 0.80 | 792 W |
| 60 s | 0.63 | 0.82 | 842 W |
| 300 s | 0.53 | 0.82 | 1090 W |
| 600 s | 0.50 | 0.83 | 1285 W |
| 900 s | 0.50 | 0.74 | 1069 W |

### L2 status

Point forecast: **gate passed** — beats both baselines at every horizon beyond
30 s by more than the fold spread. Calibration: **passed** at 30–600 s
(coverage 0.80–0.83), marginal at 900 s (0.74) where the per-fold calibration
set is small. Open items for the model ladder: a temporal-convolution / small
transformer variant evaluated on exactly this footing (and expected to *earn*
the complexity, not be assumed to); real route look-ahead features once a nav
source is available; and more calibration trips for the longest horizon.

## L3 — model-predictive control against the learned plant

`etm/control/` runs the comparison the whole project was built for: how much
heater energy does a controller need to reach the comfort BMW's production
controller actually achieved, on the same trip?

```bash
python -m etm control --processed data/processed --n-trips 8 --robustness
```

The plant is differentiable, so MPC is gradient descent through it: simulate
the planned heater schedule forward, backpropagate the cost to the plan, step,
re-plan on a receding horizon with a warm start from the previous solution.
Actions are piecewise-constant blocks — with a 44 s heater lag the plant cannot
follow anything faster, and it turns hundreds of decision variables into six.

### Comparing on a frontier, not a point

Any controller wins on energy by running the cabin cold, or on comfort by
running the heater flat out. So MPC is swept across comfort weights to trace an
energy-vs-discomfort curve, and the energy it needs **at the OEM's own comfort
level** is read off that curve by interpolation. If the frontier never reaches
the OEM's comfort, that is reported as unreachable rather than extrapolated.

Baselines it has to beat, all reported: the recorded OEM trace, a bang-bang
thermostat, and a tuned PI controller.

### Result — 7 winter trips, first 20 minutes of each

| | |
|---|---|
| Energy saved at OEM-equivalent comfort | **+23.7 % ± 6.3 %** |
| OEM heater energy over the scored span | 615 Wh mean |
| Equivalent winter range recovered | +0.62 ± 0.39 km per 20 min |

Per trip the saving ranges +13.9 % (TripB07) to +30.0 % (TripB02), positive on
all seven.

### Robustness: does it survive being wrong about the car?

MPC plans with the nominal plant and is then scored in a deliberately perturbed
one — it does not know its model is wrong, which is the actual sim-to-real gap.

| Plant error | MPC | OEM | MPC comfort |
|---|---|---|---|
| nominal | 505 Wh | 527 Wh | 33.0 vs 49.2 K·min |
| C_cab ×0.8 / ×1.2 | 460 / 552 Wh | 527 Wh | 28.2 / 37.5 vs 35.4 / 79.2 |
| UA₀ ×0.8 / ×1.2 | 475 / 536 Wh | 527 Wh | 32.7 / 33.3 vs 46.3 / 56.1 |
| η ×0.85 | 568 Wh | 527 Wh | 37.7 vs 89.6 |
| τ_h ×1.3 | 510 Wh | 527 Wh | 35.5 vs 53.1 |

MPC achieves **better comfort than the OEM under every perturbation**, at
comparable or lower energy. The advantage is not a modelling artefact that
evaporates at ±20 % parameter error.

### The caveat that matters most

**This is a simulation study.** TripB05 cannot be re-driven with a different
heater policy, so both controllers act on the L1 grey-box plant. Internal
validity is good — MPC and OEM face the identical plant, so systematic model
error largely cancels in the comparison, and the robustness sweep shows the
ranking is stable. External validity is the open question, and here is its
size, measured rather than asserted:

| Trip | OEM discomfort, simulated | …from the measured cabin trace | Cabin RMSE |
|---|---|---|---|
| B01 | 49.2 | 74.3 | 2.30 °C |
| B02 | 137.9 | 42.5 | 5.16 °C |
| B03 | 92.9 | 142.5 | 2.53 °C |
| B07 | 70.5 | 133.1 | 2.63 °C |
| B08 | 73.0 | 131.9 | 3.29 °C |
| **mean** | **79.6** | **99.2** | **2.87 °C** |

The plant's ~2.9 °C error is large next to a 1 °C comfort band, so the OEM's
*simulated* comfort differs from what it really delivered — by −63 to +96 K·min
per trip. The bias runs one way on average (simulation flatters the OEM by
20 K·min), which means MPC is held to a **stricter** comfort target than the OEM
actually achieved and the 23.7 % is, if anything, conservative. But the per-trip
spread is large, so individual trip numbers should not be quoted alone.

Two further honest limits: the OEM controller is optimising things this cost
function does not model — demisting, humidity, vent-level perceived comfort —
so "beating" it on cabin air temperature alone is not the same as beating it on
the job it was designed for. And UA₁ is still zero, so the plant has no
speed-dependent heat loss; savings on motorway trips are the least trustworthy.
