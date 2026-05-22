from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class ParameterEstimate:
    parameters: np.ndarray
    evidence: float
    log_posterior: float | None = None
    hessian: np.ndarray | None = None
    particles: np.ndarray | None = None
    weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateEstimate:
    filtered_states: np.ndarray
    covariances: np.ndarray
    next_initial_state: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowDecision:
    new_window_size: int
    metric: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DynamicalModel(ABC):
    name: str
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]

    @abstractmethod
    def simulate(
        self,
        parameters: Sequence[float],
        initial_state: np.ndarray,
        t_abs: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def linearized_step_matrix(
        self,
        parameters: Sequence[float],
        t_now: float,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError

    def default_parameters(self) -> np.ndarray:
        return np.array([(lo + hi) * 0.5 for lo, hi in self.bounds], dtype=float)

    def parameter_scale(self) -> np.ndarray:
        return np.array([max(hi - lo, 1e-6) for lo, hi in self.bounds], dtype=float)

    def sample_prior(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(lo, hi) for lo, hi in self.bounds], dtype=float)

    def clip_parameters(self, parameters: Sequence[float]) -> np.ndarray:
        clipped = np.array(parameters, dtype=float).copy()
        for index, (lo, hi) in enumerate(self.bounds):
            clipped[index] = np.clip(clipped[index], lo, hi)
        return clipped

    def initial_state_from_observations(self, observations: np.ndarray) -> np.ndarray:
        return np.array(observations[:, 0], dtype=float)

    def extract_primary_signal(self, states: np.ndarray) -> np.ndarray:
        return np.array(states[0], dtype=float)

    def log_likelihood(
        self,
        parameters: Sequence[float],
        t_abs: np.ndarray,
        observations: np.ndarray,
        initial_state: np.ndarray,
        noise_std: float,
    ) -> float:
        candidate = np.array(parameters, dtype=float)
        for value, (lo, hi) in zip(candidate, self.bounds):
            if not (lo < value < hi):
                return -1e6

        predicted = self.simulate(candidate, initial_state, t_abs)
        residual = observations - predicted
        sigma2 = max(noise_std**2, 1e-12)
        ll = -observations.size * np.log(2.0 * np.pi * sigma2)
        ll -= 0.5 * np.sum(residual**2) / sigma2
        return float(np.clip(ll, -1e6, 1e6))


class PosteriorEstimator(ABC):
    name: str

    def prepare_window(
        self,
        models: Sequence[DynamicalModel],
        t_abs: np.ndarray,
        observations: np.ndarray,
        noise_std: float,
    ) -> Mapping[str, Any]:
        return {}

    @abstractmethod
    def estimate(
        self,
        model: DynamicalModel,
        t_abs: np.ndarray,
        observations: np.ndarray,
        initial_state: np.ndarray,
        noise_std: float,
        previous_estimate: ParameterEstimate | None = None,
        shared_context: Mapping[str, Any] | None = None,
    ) -> ParameterEstimate:
        raise NotImplementedError


class StateEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(
        self,
        model: DynamicalModel,
        parameters: Sequence[float],
        t_abs: np.ndarray,
        observations: np.ndarray,
        initial_state: np.ndarray,
        dt: float,
    ) -> StateEstimate:
        raise NotImplementedError


class WindowSelector(ABC):
    name: str

    @abstractmethod
    def select(
        self,
        observed_signal: np.ndarray,
        predicted_signal: np.ndarray,
        current_window_size: int,
        previous_metric: float | None = None,
    ) -> WindowDecision:
        raise NotImplementedError
