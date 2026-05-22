from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..interfaces import DynamicalModel


def _f_state(y: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.array([y[1], -a * y[1] - b * y[0]], dtype=float)


def _trapezoidal_integrator(
    a: float,
    b_of_t: Callable[[float], float],
    initial_state: np.ndarray,
    t_abs: np.ndarray,
) -> np.ndarray:
    num_steps = len(t_abs)
    states = np.zeros((2, num_steps), dtype=float)
    states[:, 0] = np.array(initial_state, dtype=float)

    for index in range(num_steps - 1):
        step_size = float(t_abs[index + 1] - t_abs[index])
        b_now = float(b_of_t(float(t_abs[index])))
        f_now = _f_state(states[:, index], a, b_now)

        predicted = states[:, index] + step_size * f_now
        b_next = float(b_of_t(float(t_abs[index + 1])))
        f_next = _f_state(predicted, a, b_next)

        states[:, index + 1] = states[:, index] + 0.5 * step_size * (f_now + f_next)
        if not np.all(np.isfinite(states[:, index + 1])):
            raise FloatingPointError("Integration became non-finite.")

    return states


@dataclass(frozen=True)
class RegularOscillatorModel(DynamicalModel):
    name: str = "regular_free"
    parameter_names: tuple[str, ...] = ("a", "b")
    bounds: tuple[tuple[float, float], ...] = ((0.01, 0.99), (0.40, 1.60))

    def simulate(
        self,
        parameters: Sequence[float],
        initial_state: np.ndarray,
        t_abs: np.ndarray,
    ) -> np.ndarray:
        a, b = map(float, parameters)
        return _trapezoidal_integrator(a, lambda _: b, initial_state, t_abs)

    def linearized_step_matrix(
        self,
        parameters: Sequence[float],
        t_now: float,
        dt: float,
    ) -> np.ndarray:
        a, b = map(float, parameters)
        return np.array([[1.0, dt], [-b * dt, 1.0 - a * dt]], dtype=float)


@dataclass(frozen=True)
class FixedStiffnessRegularOscillatorModel(DynamicalModel):
    fixed_b: float
    name: str = "regular"
    parameter_names: tuple[str, ...] = ("a",)
    bounds: tuple[tuple[float, float], ...] = ((0.01, 0.99),)

    def simulate(
        self,
        parameters: Sequence[float],
        initial_state: np.ndarray,
        t_abs: np.ndarray,
    ) -> np.ndarray:
        a = float(parameters[0])
        return _trapezoidal_integrator(a, lambda _: self.fixed_b, initial_state, t_abs)

    def linearized_step_matrix(
        self,
        parameters: Sequence[float],
        t_now: float,
        dt: float,
    ) -> np.ndarray:
        a = float(parameters[0])
        return np.array([[1.0, dt], [-self.fixed_b * dt, 1.0 - a * dt]], dtype=float)


@dataclass(frozen=True)
class AgingOscillatorModel(DynamicalModel):
    alpha: float
    global_start_time: float
    global_total_time: float
    name: str = "aging"
    parameter_names: tuple[str, ...] = ("a", "b0")
    bounds: tuple[tuple[float, float], ...] = ((0.01, 0.99), (0.40, 1.60))

    def _stiffness(self, b0: float, t_now: float) -> float:
        total_time = max(self.global_total_time, 1e-12)
        scaled_time = (t_now - self.global_start_time) / total_time
        stiffness = b0 * (1.0 - self.alpha * scaled_time)
        return max(stiffness, 1e-4)

    def simulate(
        self,
        parameters: Sequence[float],
        initial_state: np.ndarray,
        t_abs: np.ndarray,
    ) -> np.ndarray:
        a, b0 = map(float, parameters)
        return _trapezoidal_integrator(
            a,
            lambda t_now: self._stiffness(b0, t_now),
            initial_state,
            t_abs,
        )

    def linearized_step_matrix(
        self,
        parameters: Sequence[float],
        t_now: float,
        dt: float,
    ) -> np.ndarray:
        a, b0 = map(float, parameters)
        b_eff = self._stiffness(b0, t_now)
        return np.array([[1.0, dt], [-b_eff * dt, 1.0 - a * dt]], dtype=float)
