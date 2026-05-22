from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import WindowDecision, WindowSelector


@dataclass
class AdaptiveRMSEWindowSelector(WindowSelector):
    name: str = "adaptive_rmse"
    minimum_size: int = 4
    maximum_size: int = 20
    step_size: int = 2

    def select(
        self,
        observed_signal: np.ndarray,
        predicted_signal: np.ndarray,
        current_window_size: int,
        previous_metric: float | None = None,
    ) -> WindowDecision:
        observed = np.asarray(observed_signal, dtype=float)
        predicted = np.asarray(predicted_signal, dtype=float)

        rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
        value_range = float(np.max(observed) - np.min(observed))
        metric = rmse / value_range if value_range != 0.0 else 0.0

        if previous_metric is None:
            new_window_size = current_window_size
        elif metric < previous_metric:
            new_window_size = max(self.minimum_size, current_window_size - self.step_size)
        else:
            new_window_size = min(self.maximum_size, current_window_size + self.step_size)

        return WindowDecision(
            new_window_size=int(new_window_size),
            metric=float(metric),
            metadata={"rmse": rmse},
        )
