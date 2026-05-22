from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..interfaces import DynamicalModel, StateEstimate, StateEstimator


@dataclass
class ExtendedKalmanStateEstimator(StateEstimator):
    name: str = "ekf"
    measurement_covariance_scale: float = 0.01
    process_covariance_scale: float = 0.01
    initial_covariance_scale: float = 0.1

    def estimate(
        self,
        model: DynamicalModel,
        parameters: Sequence[float],
        t_abs: np.ndarray,
        observations: np.ndarray,
        initial_state: np.ndarray,
        dt: float,
    ) -> StateEstimate:
        state_dimension, num_steps = observations.shape
        filtered_states = np.zeros((state_dimension, num_steps), dtype=float)
        covariances = np.zeros((state_dimension, state_dimension, num_steps), dtype=float)

        state = np.array(initial_state, dtype=float)
        covariance = np.eye(state_dimension, dtype=float) * self.initial_covariance_scale
        observation_matrix = np.eye(state_dimension, dtype=float)
        measurement_noise = np.eye(state_dimension, dtype=float) * self.measurement_covariance_scale
        process_noise = np.eye(state_dimension, dtype=float) * self.process_covariance_scale

        for index in range(num_steps):
            t_now = float(t_abs[min(index, len(t_abs) - 1)])
            transition = model.linearized_step_matrix(parameters, t_now, dt)
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process_noise

            innovation = observations[:, index] - observation_matrix @ predicted_state
            innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_noise
            kalman_gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)

            state = predicted_state + kalman_gain @ innovation
            covariance = (np.eye(state_dimension) - kalman_gain @ observation_matrix) @ predicted_covariance

            filtered_states[:, index] = state
            covariances[:, :, index] = covariance

        return StateEstimate(
            filtered_states=filtered_states,
            covariances=covariances,
            next_initial_state=np.array(filtered_states[:, -1], dtype=float),
        )
