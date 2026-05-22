from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelAveragingSnapshot:
    scores: dict[str, float]
    posterior: dict[str, float]


class BayesianModelAverager:
    def __init__(self, floor: float = 1e-300, cap: float = 1e300) -> None:
        self.floor = floor
        self.cap = cap

    def uniform_prior(self, model_names: list[str]) -> dict[str, float]:
        if not model_names:
            return {}
        weight = 1.0 / len(model_names)
        return {name: weight for name in model_names}

    def update(
        self,
        prior: dict[str, float],
        evidences: dict[str, float],
    ) -> ModelAveragingSnapshot:
        model_names = list(evidences.keys())
        prior_vec = np.array([prior.get(name, self.floor) for name in model_names], dtype=float)
        evidence_vec = np.array([evidences[name] for name in model_names], dtype=float)

        prior_vec = np.clip(prior_vec, self.floor, 1.0)
        evidence_vec = np.clip(evidence_vec, self.floor, self.cap)

        scores_vec = np.clip(prior_vec * evidence_vec, self.floor, self.cap)
        log_scores = np.log(prior_vec) + np.log(evidence_vec)
        log_scores -= np.max(log_scores)

        posterior_vec = np.exp(log_scores)
        posterior_vec /= np.sum(posterior_vec)

        return ModelAveragingSnapshot(
            scores={name: float(scores_vec[index]) for index, name in enumerate(model_names)},
            posterior={name: float(posterior_vec[index]) for index, name in enumerate(model_names)},
        )
