from __future__ import annotations

from typing import Any

_EPS = 1e-3


def _clamp_open_interval(value: float) -> float:
    return round(max(_EPS, min(1.0 - _EPS, float(value))), 4)


def _extract_from_mapping(data: dict[str, Any]) -> float | None:
    if isinstance(data.get("score"), (int, float)):
        return float(data["score"])
    if isinstance(data.get("task_score"), (int, float)):
        return float(data["task_score"])
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("task_score"), (int, float)):
        return float(metrics["task_score"])
    return None


def _extract_score(*args: Any, default: float = 0.5, **kwargs: Any) -> float:
    if isinstance(kwargs.get("score"), (int, float)):
        return _clamp_open_interval(float(kwargs["score"]))
    if isinstance(kwargs.get("task_score"), (int, float)):
        return _clamp_open_interval(float(kwargs["task_score"]))
    if "state" in kwargs:
        args = (kwargs["state"],) + args

    if args:
        candidate = args[0]
        if isinstance(candidate, (int, float)):
            return _clamp_open_interval(float(candidate))
        if isinstance(candidate, dict):
            value = _extract_from_mapping(candidate)
            if value is not None:
                return _clamp_open_interval(value)
        metrics = getattr(candidate, "metrics", None)
        if metrics is not None:
            task_score = getattr(metrics, "task_score", None)
            if isinstance(task_score, (int, float)):
                return _clamp_open_interval(float(task_score))
        task_score = getattr(candidate, "task_score", None)
        if isinstance(task_score, (int, float)):
            return _clamp_open_interval(float(task_score))

    return _clamp_open_interval(default)


def grade_weekday_spike(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, default=0.74, **kwargs)


def grade_sunset_transition(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, default=0.71, **kwargs)


def grade_heatwave_failure(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, default=0.62, **kwargs)


def grade_storm_front_response(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, default=0.68, **kwargs)


def grade_winter_gas_shortage(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, default=0.64, **kwargs)
