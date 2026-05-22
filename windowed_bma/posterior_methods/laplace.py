from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..interfaces import DynamicalModel, ParameterEstimate, PosteriorEstimator


@dataclass
class LaplacePosteriorEstimator(PosteriorEstimator):
    name: str = "laplace"
    finite_difference_step: float = 1e-4
    prior_precision: float | Sequence[float] | None = None
    search_iterations: int = 80
    random_starts: int = 96
    random_seed: int = 42

    def _resolve_prior_precision(self, dimension: int) -> np.ndarray | None:
        if self.prior_precision is None:
            return None
        if np.isscalar(self.prior_precision):
            return np.full(dimension, float(self.prior_precision), dtype=float)
        values = np.array(self.prior_precision, dtype=float)
        if len(values) != dimension:
            raise ValueError("prior_precision size must match the parameter dimension")
        return values

    def estimate(
        self,
        model: DynamicalModel,
        t_abs: np.ndarray,
        observations: np.ndarray,
        initial_state: np.ndarray,
        noise_std: float,
        previous_estimate: ParameterEstimate | None = None,
        shared_context: Mapping[str, object] | None = None,
    ) -> ParameterEstimate:
        del shared_context

        initial_guess = (
            np.array(previous_estimate.parameters, dtype=float)
            if previous_estimate is not None
            else model.default_parameters()
        )
        dimension = len(initial_guess)
        prior_center = (
            np.array(previous_estimate.parameters, dtype=float)
            if previous_estimate is not None and self.prior_precision is not None
            else None
        )
        prior_precision = self._resolve_prior_precision(dimension)

        def neg_log_post(theta: np.ndarray) -> float:
            nlp = -model.log_likelihood(theta, t_abs, observations, initial_state, noise_std)
            if prior_center is not None and prior_precision is not None:
                nlp += 0.5 * np.sum(prior_precision * (theta - prior_center) ** 2)
            return float(nlp)

        search_rng = np.random.default_rng(self.random_seed)

        def bounded_search(x0: np.ndarray) -> tuple[np.ndarray, float, bool]:
            best_theta = model.clip_parameters(x0)
            best_value = neg_log_post(best_theta)

            for _ in range(self.random_starts):
                candidate = model.sample_prior(search_rng)
                value = neg_log_post(candidate)
                if value < best_value:
                    best_theta = candidate
                    best_value = value

            step_sizes = 0.25 * model.parameter_scale()
            success = True
            for _ in range(self.search_iterations):
                improved = False
                for index, (lo, hi) in enumerate(model.bounds):
                    for direction in (-1.0, 1.0):
                        candidate = best_theta.copy()
                        candidate[index] = np.clip(
                            candidate[index] + direction * step_sizes[index],
                            lo,
                            hi,
                        )
                        value = neg_log_post(candidate)
                        if value < best_value:
                            best_theta = candidate
                            best_value = value
                            improved = True
                if not improved:
                    step_sizes *= 0.5
                if np.max(step_sizes) < 1e-5:
                    break

            if not np.all(np.isfinite(best_theta)):
                success = False
                best_theta = model.clip_parameters(x0)
                best_value = neg_log_post(best_theta)
            return best_theta, float(best_value), success

        theta_map, _, success = bounded_search(np.array(initial_guess, dtype=float))

        step = self.finite_difference_step
        hessian_diag = np.zeros(dimension, dtype=float)
        for index in range(dimension):
            direction = np.zeros(dimension, dtype=float)
            direction[index] = step
            fpp = (
                neg_log_post(theta_map + direction)
                - 2.0 * neg_log_post(theta_map)
                + neg_log_post(theta_map - direction)
            ) / (step**2)
            hessian_diag[index] = max(float(fpp), 1e-6)

        hessian = np.diag(hessian_diag)
        log_post = -neg_log_post(theta_map)
        normalizer = np.sqrt(np.linalg.det(hessian)) + 1e-12
        evidence = np.exp(np.clip(log_post, -700.0, 700.0))
        evidence *= (2.0 * np.pi) ** (dimension / 2.0)
        evidence /= normalizer
        evidence = float(np.clip(evidence, 1e-300, 1e300))

        return ParameterEstimate(
            parameters=np.array(theta_map, dtype=float),
            evidence=evidence,
            log_posterior=float(log_post),
            hessian=hessian,
            metadata={
                "success": bool(success),
                "message": "bounded coordinate search",
            },
        )
