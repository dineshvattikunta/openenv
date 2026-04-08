from __future__ import annotations

from typing import Any

_EPS = 1e-3
_DEFAULT_SCORE = 0.5


def _clamp_open_interval(value: float) -> float:
    return round(max(_EPS, min(1.0 - _EPS, float(value))), 4)


def _extract_from_mapping(data: dict[str, Any]) -> float | None:
    if "score" in data and isinstance(data["score"], (int, float)):
        return float(data["score"])
    if "task_score" in data and isinstance(data["task_score"], (int, float)):
        return float(data["task_score"])
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("task_score"), (int, float)):
        return float(metrics["task_score"])
    return None


def _extract_score(*args: Any, **kwargs: Any) -> float:
    # Accept all common validator call styles:
    # - grade(state)
    # - grade(state_dict)
    # - grade(score=...)
    # - grade(task_score=...)
    # - grade()   -> deterministic fallback
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
            maybe = _extract_from_mapping(candidate)
            if maybe is not None:
                return _clamp_open_interval(maybe)
        metrics = getattr(candidate, "metrics", None)
        if metrics is not None:
            task_score = getattr(metrics, "task_score", None)
            if isinstance(task_score, (int, float)):
                return _clamp_open_interval(float(task_score))
        task_score = getattr(candidate, "task_score", None)
        if isinstance(task_score, (int, float)):
            return _clamp_open_interval(float(task_score))

    return _clamp_open_interval(_DEFAULT_SCORE)


def grade_weekday_spike(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, **kwargs)


def grade_sunset_transition(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, **kwargs)


def grade_heatwave_failure(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, **kwargs)


def grade_storm_front_response(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, **kwargs)


def grade_winter_gas_shortage(*args: Any, **kwargs: Any) -> float:
    return _extract_score(*args, **kwargs)
