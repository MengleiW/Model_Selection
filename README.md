# Generalized Windowed Model Averaging

This refactor turns the original single-file script into a modular workflow where each major method can be swapped independently.

## Structure

- `main.py`
  - prepares input data
  - chooses models
  - chooses posterior estimators
  - chooses state estimator
  - chooses window-size rule
  - runs the full workflow
- `windowed_bma/models/`
  - model definitions live here
- `windowed_bma/posterior_methods/`
  - posterior / parameter estimation methods live here
- `windowed_bma/state_estimators/`
  - state estimation methods live here
- `windowed_bma/window_selection/`
  - adaptive window rules live here
- `windowed_bma/orchestrator.py`
  - the shared runner that connects everything
- `windowed_bma/model_averaging.py`
  - Bayesian updating of model probabilities across windows

## Included example components

- Models
  - `FixedStiffnessRegularOscillatorModel`
  - `AgingOscillatorModel`
  - `RegularOscillatorModel` as an extra free-parameter variant
- Posterior methods
  - `LaplacePosteriorEstimator`
  - `ABCSMCPosteriorEstimator`
- State estimator
  - `ExtendedKalmanStateEstimator`
- Window selector
  - `AdaptiveRMSEWindowSelector`

## How to add a new model

Create a new file or class under `windowed_bma/models/` that inherits from `DynamicalModel` and implements:

- `simulate(...)`
- `linearized_step_matrix(...)`

You can also override:

- `sample_prior(...)`
- `default_parameters(...)`
- `log_likelihood(...)`

## How to add a new posterior method

Create a class under `windowed_bma/posterior_methods/` that inherits from `PosteriorEstimator` and implements:

- `estimate(...)`

Optional:

- `prepare_window(...)` if the method needs shared per-window setup, such as a joint epsilon schedule.

## How to add a new state estimator

Create a class under `windowed_bma/state_estimators/` that inherits from `StateEstimator` and implements:

- `estimate(...)`

## How to add a new window rule

Create a class under `windowed_bma/window_selection/` that inherits from `WindowSelector` and implements:

- `select(...)`

## Running the example

```powershell
python main.py
```

The example uses your oscillator setup as the first test case and writes plots to `outputs/`.
