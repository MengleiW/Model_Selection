from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..interfaces import DynamicalModel, ParameterEstimate, PosteriorEstimator


@dataclass
class ABCSMCPosteriorEstimator(PosteriorEstimator):
    name: str = "abc_smc"
    num_particles: int = 200
    num_generations: int = 3
    schedule_num_samples: int = 300
    schedule_quantiles: tuple[float, ...] = (60.0, 40.0, 25.0)
    max_total_attempts: int = 150_000
    minimum_epsilon: float = 0.05
    kernel_scale: float = 0.10
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def distance(
        self,
        observations: np.ndarray,
        simulated: np.ndarray,
        noise_std: float,
    ) -> float:
        sigma = max(noise_std, 1e-12)
        residual = (observations - simulated) / sigma
        return float(np.sqrt(np.mean(residual**2)))

    def prepare_window(
        self,
        models,
        t_abs: np.ndarray,
        observations: np.ndarray,
        noise_std: float,
    ) -> Mapping[str, object]:
        distances: list[float] = []
        if not models:
            return {"eps_schedule": np.array([1.0, 0.5, 0.2], dtype=float)}

        samples_per_model = max(1, self.schedule_num_samples // len(models))
        remainder = self.schedule_num_samples - samples_per_model * len(models)

        for index, model in enumerate(models):
            sample_count = samples_per_model + (1 if index < remainder else 0)
            initial_state = model.initial_state_from_observations(observations)
            for _ in range(sample_count):
                parameters = model.sample_prior(self.rng)
                simulated = model.simulate(parameters, initial_state, t_abs)
                distance = self.distance(observations, simulated, noise_std)
                if np.isfinite(distance):
                    distances.append(distance)

        if len(distances) < 10:
            eps_schedule = np.array([1.0, 0.5, 0.2], dtype=float)
        else:
            eps_schedule = np.percentile(np.array(distances, dtype=float), self.schedule_quantiles)
        return {"eps_schedule": np.maximum(eps_schedule, self.minimum_epsilon)}

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
        dimension = len(model.parameter_names)
        eps_schedule = None
        if shared_context is not None and "eps_schedule" in shared_context:
            eps_schedule = np.array(shared_context["eps_schedule"], dtype=float)
        if eps_schedule is None:
            eps_schedule = np.array([1.0, 0.5, 0.2], dtype=float)

        particles = np.zeros((self.num_generations, self.num_particles, dimension), dtype=float)
        weights = np.zeros((self.num_generations, self.num_particles), dtype=float)
        accepted_counts = np.zeros(self.num_generations, dtype=int)

        prior_particles = None
        prior_weights = None
        if previous_estimate is not None and previous_estimate.particles is not None:
            prior_particles = np.atleast_2d(np.array(previous_estimate.particles, dtype=float))
            if previous_estimate.weights is not None and len(previous_estimate.weights) == len(prior_particles):
                prior_weights = np.array(previous_estimate.weights, dtype=float)
                prior_weights = prior_weights / max(np.sum(prior_weights), 1e-12)

        total_attempts = 0
        previous_count = 0

        for generation in range(self.num_generations):
            epsilon = float(eps_schedule[min(generation, len(eps_schedule) - 1)])
            accepted = 0
            use_uniform = False
            sigma = self.kernel_scale * model.parameter_scale()

            if generation > 0:
                previous_particles = particles[generation - 1, :previous_count, :]
                previous_weights = weights[generation - 1, :previous_count]
                if previous_count == 0 or np.sum(previous_weights) <= 1e-12 or not np.isfinite(previous_weights).all():
                    use_uniform = True
                else:
                    previous_weights = previous_weights / np.sum(previous_weights)
                    mean = np.average(previous_particles, axis=0, weights=previous_weights)
                    variance = np.average((previous_particles - mean) ** 2, axis=0, weights=previous_weights)
                    sigma = self.kernel_scale * np.maximum(np.sqrt(variance), 1e-6)

            while accepted < self.num_particles and total_attempts < self.max_total_attempts:
                total_attempts += 1
                if generation == 0 or use_uniform:
                    if prior_particles is not None and prior_weights is not None:
                        choice = self.rng.choice(len(prior_particles), p=prior_weights)
                        parameters = np.array(prior_particles[choice], dtype=float)
                    else:
                        parameters = model.sample_prior(self.rng)
                else:
                    previous_particles = particles[generation - 1, :previous_count, :]
                    previous_weights = weights[generation - 1, :previous_count]
                    previous_weights = previous_weights / np.sum(previous_weights)
                    choice = self.rng.choice(previous_count, p=previous_weights)
                    center = previous_particles[choice]
                    parameters = model.clip_parameters(center + self.rng.normal(0.0, sigma, size=dimension))

                simulated = model.simulate(parameters, initial_state, t_abs)
                distance = self.distance(observations, simulated, noise_std)
                if distance <= epsilon:
                    particles[generation, accepted, :] = parameters
                    accepted += 1

            previous_count = accepted
            accepted_counts[generation] = accepted

            if generation == 0 or use_uniform:
                weights[generation, :accepted] = 1.0 / max(accepted, 1)
                continue

            previous_particles = particles[generation - 1, :previous_count, :]
            previous_weights = weights[generation - 1, :previous_count]
            if previous_count == 0 or np.sum(previous_weights) <= 1e-12 or not np.isfinite(previous_weights).all():
                weights[generation, :accepted] = 1.0 / max(accepted, 1)
                continue

            previous_weights = previous_weights / np.sum(previous_weights)
            new_weights = np.zeros(accepted, dtype=float)
            for accepted_index in range(accepted):
                diffs = (particles[generation, accepted_index, :] - previous_particles) / np.maximum(sigma, 1e-12)
                kernel = np.exp(-0.5 * np.sum(diffs**2, axis=1))
                denominator = np.sum(previous_weights * kernel)
                new_weights[accepted_index] = 1.0 / max(denominator, 1e-12)

            new_weights /= max(np.sum(new_weights), 1e-12)
            weights[generation, :accepted] = new_weights

        valid_generations = np.where(accepted_counts > 0)[0]
        if valid_generations.size == 0:
            return ParameterEstimate(
                parameters=model.default_parameters(),
                evidence=1e-12,
                particles=np.empty((0, dimension), dtype=float),
                weights=np.empty((0,), dtype=float),
                metadata={"accepted_counts": accepted_counts.tolist()},
            )

        generation_index = int(valid_generations[-1])
        used_count = int(accepted_counts[generation_index])
        last_particles = particles[generation_index, :used_count, :]
        last_weights = weights[generation_index, :used_count]
        if np.sum(last_weights) <= 1e-12 or not np.isfinite(last_weights).all():
            last_weights = np.ones(len(last_weights), dtype=float) / max(len(last_weights), 1)
        else:
            last_weights = last_weights / np.sum(last_weights)

        parameter_mean = np.average(last_particles, axis=0, weights=last_weights)
        last_epsilon = float(eps_schedule[min(generation_index, len(eps_schedule) - 1)])
        evidence = (accepted_counts[generation_index] / max(total_attempts, 1)) / max(last_epsilon, 1e-12)
        evidence = float(np.clip(evidence, 1e-300, 1e300))

        return ParameterEstimate(
            parameters=np.array(parameter_mean, dtype=float),
            evidence=evidence,
            particles=np.array(last_particles, dtype=float),
            weights=np.array(last_weights, dtype=float),
            metadata={
                "eps_schedule": np.array(eps_schedule, dtype=float).tolist(),
                "accepted_counts": accepted_counts.tolist(),
                "total_attempts": int(total_attempts),
            },
        )
