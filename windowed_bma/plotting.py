from __future__ import annotations

from pathlib import Path

import numpy as np

from .results import AnalysisResult


def _build_piecewise_series(
    result: AnalysisResult,
    method_name: str,
    value_getter,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not result.windows:
        return np.array([], dtype=float), {}

    model_names = list(result.windows[0].method_results[method_name].posterior_by_model.keys())
    times_piecewise: list[float] = []
    values = {name: [] for name in model_names}

    for window in result.windows:
        method_result = window.method_results[method_name]
        segment_times = result.times[window.start_index:window.end_index]
        for time_value in segment_times:
            times_piecewise.append(float(time_value))
        for model_name in model_names:
            segment_value = value_getter(method_result, model_name)
            values[model_name].extend([segment_value] * len(segment_times))

    if times_piecewise:
        times_piecewise.append(times_piecewise[-1] + 1e-12)
        for series in values.values():
            series.append(series[-1])

    return np.array(times_piecewise, dtype=float), {
        name: np.array(series, dtype=float) for name, series in values.items()
    }


def _add_boundary_markers(ax, boundary_times: np.ndarray, log_scale: bool = False) -> None:
    if len(boundary_times) == 0:
        return
    ymin, ymax = ax.get_ylim()
    if log_scale:
        marker_y = ymin * 1.05
    else:
        marker_y = ymin + 0.02 * (ymax - ymin)
    for boundary_time in boundary_times:
        ax.plot(boundary_time, marker_y, "o", ms=5, mfc="white", mec="k", zorder=5)
    ax.set_ylim(ymin, ymax)


def _should_show_plots(plt_module) -> bool:
    backend = str(plt_module.get_backend()).lower()
    return "agg" not in backend


def plot_method_scores(
    result: AnalysisResult,
    method_name: str,
    output_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    time_series, evidence_values = _build_piecewise_series(
        result,
        method_name,
        lambda method_result, model_name: method_result.evidences_by_model[model_name],
    )
    _, posterior_values = _build_piecewise_series(
        result,
        method_name,
        lambda method_result, model_name: method_result.posterior_by_model[model_name],
    )
    if len(time_series) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for model_name, series in evidence_values.items():
        ax1.step(time_series, np.clip(series, 1e-300, 1e300), where="post", label=model_name)
    ax1.set_yscale("log")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Evidence")
    ax1.set_title(f"{method_name} evidence")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    _add_boundary_markers(ax1, result.boundary_times, log_scale=True)

    for model_name, series in posterior_values.items():
        ax2.step(time_series, series, where="post", label=f"{model_name} posterior")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Model posterior")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title(f"{method_name} model posterior")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _add_boundary_markers(ax2, result.boundary_times, log_scale=False)

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if _should_show_plots(plt):
        plt.show()
    else:
        plt.close(fig)


def plot_method_predictions(
    result: AnalysisResult,
    method_name: str,
    observed_signal: np.ndarray,
    true_signal: np.ndarray | None = None,
    output_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    if true_signal is not None:
        plt.plot(result.times, true_signal, "g-", lw=2, label="True")
    plt.plot(result.times, observed_signal, "k.", ms=4, alpha=0.4, label="Measurements")

    for model_name, series in result.model_predictions[method_name].items():
        plt.plot(result.times, series, "--", lw=2, label=f"{model_name} prediction")

    plt.plot(
        result.times,
        result.weighted_predictions[method_name],
        "m-",
        lw=2,
        label=f"{method_name} weighted average",
    )

    if len(result.boundary_times) > 0:
        ymin, ymax = plt.ylim()
        marker_y = ymin + 0.02 * (ymax - ymin)
        for boundary_time in result.boundary_times:
            plt.plot(boundary_time, marker_y, "o", ms=6, mfc="white", mec="k", zorder=5)
        plt.ylim(ymin, ymax)

    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title(f"Trajectories using {method_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if _should_show_plots(plt):
        plt.show()
    else:
        plt.close()
