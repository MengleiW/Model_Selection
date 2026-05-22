from __future__ import annotations

from pathlib import Path

import numpy as np

from windowed_bma import RunConfig, WindowConfig, WindowedModelAveragingRunner
from windowed_bma.models import AgingOscillatorModel, FixedStiffnessRegularOscillatorModel
from windowed_bma.posterior_methods import ABCSMCPosteriorEstimator, LaplacePosteriorEstimator
from windowed_bma.state_estimators import ExtendedKalmanStateEstimator
from windowed_bma.window_selection import AdaptiveRMSEWindowSelector

try:
    from windowed_bma.plotting import plot_method_predictions, plot_method_scores
except ModuleNotFoundError:
    plot_method_predictions = None
    plot_method_scores = None


ALPHA_AGE = 0.45
NOISE_STD = 0.05
DT = 0.5
T_TOTAL = 15.0
WIN_SEC0 = 3.0
W_MIN, W_MAX, DELTA_S = 4, 20, 2
RNG = np.random.default_rng(42)


def build_example_data(
    truth_model: str = "aging",
    a_true: float = 0.30,
    b_true0: float = 1.00,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, T_TOTAL + DT, DT)
    regular_model = FixedStiffnessRegularOscillatorModel(fixed_b=b_true0)
    aging_model = AgingOscillatorModel(
        alpha=ALPHA_AGE,
        global_start_time=float(times[0]),
        global_total_time=float(times[-1] - times[0]),
    )

    if truth_model == "regular":
        true_states = regular_model.simulate([a_true], np.array([1.0, 0.0]), times)
    else:
        true_states = aging_model.simulate([a_true, b_true0], np.array([1.0, 0.0]), times)

    noisy_observations = true_states + RNG.normal(0.0, NOISE_STD, size=true_states.shape)
    return times, noisy_observations, true_states


def main() -> None:
    times, observations, true_states = build_example_data()

    regular_model = FixedStiffnessRegularOscillatorModel(fixed_b=1.00)
    aging_model = AgingOscillatorModel(
        alpha=ALPHA_AGE,
        global_start_time=float(times[0]),
        global_total_time=float(times[-1] - times[0]),
    )

    posterior_estimators = [
        LaplacePosteriorEstimator(name="laplace"),
        ABCSMCPosteriorEstimator(name="abc_smc"),
    ]
    state_estimator = ExtendedKalmanStateEstimator()
    window_selector = AdaptiveRMSEWindowSelector(
        minimum_size=W_MIN,
        maximum_size=W_MAX,
        step_size=DELTA_S,
    )
    config = RunConfig(
        dt=DT,
        noise_std=NOISE_STD,
        window=WindowConfig(
            initial_size=int(round(WIN_SEC0 / DT)),
            minimum_size=W_MIN,
            maximum_size=W_MAX,
            step_size=DELTA_S,
        ),
        state_estimation_source="laplace",
        window_decision_source="abc_smc",
    )

    runner = WindowedModelAveragingRunner(
        config=config,
        models=[regular_model, aging_model],
        posterior_estimators=posterior_estimators,
        state_estimator=state_estimator,
        window_selector=window_selector,
    )
    result = runner.run(times, observations)

    output_dir = Path("outputs")
    plots_created = False
    if plot_method_scores is not None and plot_method_predictions is not None:
        try:
            plot_method_scores(result, "laplace", output_dir / "laplace_scores.png")
            plot_method_predictions(
                result,
                "laplace",
                observed_signal=observations[0],
                true_signal=true_states[0],
                output_path=output_dir / "laplace_predictions.png",
            )
            plot_method_scores(result, "abc_smc", output_dir / "abc_scores.png")
            plot_method_predictions(
                result,
                "abc_smc",
                observed_signal=observations[0],
                true_signal=true_states[0],
                output_path=output_dir / "abc_predictions.png",
            )
            plots_created = True
        except ModuleNotFoundError:
            plots_created = False

    print("Windowed analysis complete.")
    if not plots_created:
        print("matplotlib is not installed in this runtime, so plot generation was skipped.")
    else:
        print("Saved plots to:", output_dir.resolve())
    print("Windows processed:", len(result.windows))
    for method_name in ("laplace", "abc_smc"):
        final_posterior = result.windows[-1].method_results[method_name].posterior_by_model
        print(f"Final posterior for {method_name}: {final_posterior}")


if __name__ == "__main__":
    main()
