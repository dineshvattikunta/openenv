from __future__ import annotations

from typing import Any, Callable, Dict

from env.models import GridState
from env.rewards import SCORE_EPSILON


def clamp_score(value: float) -> float:
    value = float(value)
    value = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, value))
    value = round(value, 2)
    value = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, value))
    return value


def _extract_from_mapping(data: dict[str, Any]) -> float | None:
    if isinstance(data.get("score"), (int, float)):
        return float(data["score"])
    if isinstance(data.get("task_score"), (int, float)):
        return float(data["task_score"])
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("task_score"), (int, float)):
        return float(metrics["task_score"])
    return None


def _extract_score_and_state(*args: Any, default: float = 0.5, **kwargs: Any) -> tuple[float, GridState | None]:
    if "state" in kwargs and args:
        args = (kwargs["state"],) + args
    elif "state" in kwargs:
        args = (kwargs["state"],)

    if isinstance(kwargs.get("score"), (int, float)):
        return clamp_score(float(kwargs["score"])), (kwargs.get("state") if isinstance(kwargs.get("state"), GridState) else None)
    if isinstance(kwargs.get("task_score"), (int, float)):
        return clamp_score(float(kwargs["task_score"])), (kwargs.get("state") if isinstance(kwargs.get("state"), GridState) else None)

    if args:
        candidate = args[0]
        if isinstance(candidate, GridState):
            return clamp_score(candidate.metrics.task_score), candidate
        if isinstance(candidate, (int, float)):
            return clamp_score(float(candidate)), None
        if isinstance(candidate, dict):
            maybe = _extract_from_mapping(candidate)
            if maybe is not None:
                return clamp_score(maybe), None
        metrics = getattr(candidate, "metrics", None)
        if metrics is not None:
            task_score = getattr(metrics, "task_score", None)
            if isinstance(task_score, (int, float)):
                return clamp_score(float(task_score)), (candidate if isinstance(candidate, GridState) else None)
        task_score = getattr(candidate, "task_score", None)
        if isinstance(task_score, (int, float)):
            return clamp_score(float(task_score)), None

    return clamp_score(default), None


def task_grade_breakdown(state: GridState | None = None, *, score: float | None = None) -> Dict[str, float]:
    if state is None:
        base = clamp_score(score if score is not None else 0.5)
        return {
            "served_ratio": base,
            "critical_served_ratio": base,
            "efficiency_ratio": base,
            "cost_efficiency_ratio": base,
            "response_ratio": base,
        }

    metrics = state.metrics

    return {
        "served_ratio": clamp_score(metrics.average_served_ratio),
        "critical_served_ratio": clamp_score(metrics.average_critical_served_ratio),
        "efficiency_ratio": clamp_score(metrics.average_efficiency_ratio),
        "cost_efficiency_ratio": clamp_score(1.0 - metrics.average_cost_ratio),
        "response_ratio": clamp_score(metrics.average_response_ratio),
    }


def grade_task(*args: Any, **kwargs: Any) -> float:
    score, _ = _extract_score_and_state(*args, **kwargs)
    return score


def grade_report(task_id: str, state: GridState | None = None, *args: Any, **kwargs: Any) -> Dict[str, object]:
    score, extracted_state = _extract_score_and_state(state, *args, **kwargs)
    chosen_state = state if isinstance(state, GridState) else extracted_state
    return {
        "task_id": task_id,
        "score": score,
        "breakdown": task_grade_breakdown(chosen_state, score=score),
    }


def grade_weekday_spike(*args: Any, **kwargs: Any) -> float:
    return grade_task(*args, **kwargs)


def grade_sunset_transition(*args: Any, **kwargs: Any) -> float:
    return grade_task(*args, **kwargs)


def grade_heatwave_failure(*args: Any, **kwargs: Any) -> float:
    return grade_task(*args, **kwargs)


def grade_storm_front_response(*args: Any, **kwargs: Any) -> float:
    return grade_task(*args, **kwargs)


def grade_winter_gas_shortage(*args: Any, **kwargs: Any) -> float:
    return grade_task(*args, **kwargs)


GRADERS: Dict[str, Callable[..., float]] = {
    "weekday_spike": grade_weekday_spike,
    "sunset_transition": grade_sunset_transition,
    "heatwave_failure": grade_heatwave_failure,
    "storm_front_response": grade_storm_front_response,
    "winter_gas_shortage": grade_winter_gas_shortage,
}
