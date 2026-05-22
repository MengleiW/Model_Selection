from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import ParameterEstimate


@dataclass
class MethodWindowResult:
    method_name: str
    estimates_by_model: dict[str, ParameterEstimate]
    evidences_by_model: dict[str, float]
    scores_by_model: dict[str, float]
    posterior_by_model: dict[str, float]
    weighted_prediction: np.ndarray


@dataclass
class WindowRecord:
    start_index: int
    end_index: int
    window_size: int
    decision_metric: float | None
    method_results: dict[str, MethodWindowResult]


@dataclass
class AnalysisResult:
    times: np.ndarray
    observations: np.ndarray
    windows: list[WindowRecord]
    model_predictions: dict[str, dict[str, np.ndarray]]
    weighted_predictions: dict[str, np.ndarray]
    boundary_times: np.ndarray
