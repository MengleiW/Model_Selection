from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import RunConfig
from .interfaces import DynamicalModel, ParameterEstimate, PosteriorEstimator, StateEstimator, WindowSelector
from .model_averaging import BayesianModelAverager
from .results import AnalysisResult, MethodWindowResult, WindowRecord


@dataclass
class WindowedModelAveragingRunner:
    config: RunConfig
    models: Sequence[DynamicalModel]
    posterior_estimators: Sequence[PosteriorEstimator]
    state_estimator: StateEstimator
    window_selector: WindowSelector
    averager: BayesianModelAverager = BayesianModelAverager()

    def run(self, times: np.ndarray, observations: np.ndarray) -> AnalysisResult:
        if observations.ndim != 2:
            raise ValueError("observations must have shape (state_dimension, num_times)")
        if observations.shape[1] != len(times):
            raise ValueError("times and observations must have matching lengths")

        model_names = [model.name for model in self.models]
        method_names = [method.name for method in self.posterior_estimators]

        model_predictions = {
            method_name: {model_name: np.zeros(len(times), dtype=float) for model_name in model_names}
            for method_name in method_names
        }
        weighted_predictions = {
            method_name: np.zeros(len(times), dtype=float)
            for method_name in method_names
        }

        previous_estimates: dict[str, dict[str, ParameterEstimate | None]] = {
            method_name: {model_name: None for model_name in model_names}
            for method_name in method_names
        }
        model_priors: dict[str, dict[str, float]] = {
            method_name: self.averager.uniform_prior(model_names)
            for method_name in method_names
        }
        next_initial_states: dict[str, np.ndarray | None] = {
            model_name: None for model_name in model_names
        }

        current_window_size = self.config.window.initial_size
        previous_metric: float | None = None
        start_index = 0
        windows: list[WindowRecord] = []
        boundary_times: list[float] = []

        while start_index + 2 <= len(times):
            if start_index + current_window_size + 1 > len(times):
                current_window_size = (len(times) - 1) - start_index
            end_index = start_index + current_window_size + 1
            if end_index <= start_index + 2:
                break

            boundary_times.append(float(times[start_index]))
            t_window = times[start_index:end_index]
            obs_window = observations[:, start_index:end_index]

            method_results: dict[str, MethodWindowResult] = {}
            source_predictions: dict[str, np.ndarray] = {}

            for estimator in self.posterior_estimators:
                shared_context = estimator.prepare_window(
                    self.models,
                    t_window,
                    obs_window,
                    self.config.noise_std,
                )

                estimates_by_model: dict[str, ParameterEstimate] = {}
                evidences_by_model: dict[str, float] = {}
                signals_by_model: dict[str, np.ndarray] = {}

                for model in self.models:
                    initial_state = next_initial_states[model.name]
                    if initial_state is None:
                        initial_state = model.initial_state_from_observations(obs_window)

                    estimate = estimator.estimate(
                        model=model,
                        t_abs=t_window,
                        observations=obs_window,
                        initial_state=np.array(initial_state, dtype=float),
                        noise_std=self.config.noise_std,
                        previous_estimate=previous_estimates[estimator.name][model.name],
                        shared_context=shared_context,
                    )
                    previous_estimates[estimator.name][model.name] = estimate

                    simulation = model.simulate(estimate.parameters, np.array(initial_state, dtype=float), t_window)
                    signal = model.extract_primary_signal(simulation)

                    estimates_by_model[model.name] = estimate
                    evidences_by_model[model.name] = estimate.evidence
                    signals_by_model[model.name] = signal
                    model_predictions[estimator.name][model.name][start_index:end_index] = signal

                averaging_snapshot = self.averager.update(model_priors[estimator.name], evidences_by_model)
                model_priors[estimator.name] = averaging_snapshot.posterior

                weighted_prediction = np.zeros(end_index - start_index, dtype=float)
                for model_name, signal in signals_by_model.items():
                    weighted_prediction += averaging_snapshot.posterior[model_name] * signal

                weighted_predictions[estimator.name][start_index:end_index] = weighted_prediction
                source_predictions[estimator.name] = weighted_prediction

                method_results[estimator.name] = MethodWindowResult(
                    method_name=estimator.name,
                    estimates_by_model=estimates_by_model,
                    evidences_by_model=evidences_by_model,
                    scores_by_model=averaging_snapshot.scores,
                    posterior_by_model=averaging_snapshot.posterior,
                    weighted_prediction=weighted_prediction,
                )

            source_method = self.config.state_estimation_source
            if source_method not in method_results:
                raise KeyError(f"Unknown state estimation source method: {source_method}")

            for model in self.models:
                initial_state = next_initial_states[model.name]
                if initial_state is None:
                    initial_state = model.initial_state_from_observations(obs_window)

                estimate = method_results[source_method].estimates_by_model[model.name]
                state_result = self.state_estimator.estimate(
                    model=model,
                    parameters=estimate.parameters,
                    t_abs=t_window,
                    observations=obs_window,
                    initial_state=np.array(initial_state, dtype=float),
                    dt=self.config.dt,
                )
                next_initial_states[model.name] = state_result.next_initial_state

            decision_source = self.config.window_decision_source
            if decision_source not in source_predictions:
                raise KeyError(f"Unknown window decision source method: {decision_source}")

            window_decision = self.window_selector.select(
                observed_signal=obs_window[0],
                predicted_signal=source_predictions[decision_source],
                current_window_size=current_window_size,
                previous_metric=previous_metric,
            )

            previous_metric = window_decision.metric
            windows.append(
                WindowRecord(
                    start_index=start_index,
                    end_index=end_index,
                    window_size=current_window_size,
                    decision_metric=window_decision.metric,
                    method_results=method_results,
                )
            )

            current_window_size = window_decision.new_window_size
            start_index = end_index - 1

        if windows:
            boundary_times.append(float(times[windows[-1].end_index - 1]))

        return AnalysisResult(
            times=np.array(times, dtype=float),
            observations=np.array(observations, dtype=float),
            windows=windows,
            model_predictions=model_predictions,
            weighted_predictions=weighted_predictions,
            boundary_times=np.array(boundary_times, dtype=float),
        )
