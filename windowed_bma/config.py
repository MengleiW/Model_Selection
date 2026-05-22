from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    initial_size: int
    minimum_size: int
    maximum_size: int
    step_size: int


@dataclass(frozen=True)
class RunConfig:
    dt: float
    noise_std: float
    window: WindowConfig
    state_estimation_source: str
    window_decision_source: str
