from __future__ import annotations

from typing import Callable, Dict

from env.models import GridState
from env.rewards import SCORE_EPSILON
from env.utils import clamp


def task_grade_breakdown(state: GridState) -> Dict[str, float]:
    metrics = state.metrics
    def safe(value: float) -> float:
        return round(clamp(value, SCORE_EPSILON, 1.0 - SCORE_EPSILON), 4)

    return {
        "served_ratio": safe(metrics.average_served_ratio),
        "critical_served_ratio": safe(metrics.average_critical_served_ratio),
        "efficiency_ratio": safe(metrics.average_efficiency_ratio),
        "cost_efficiency_ratio": safe(1.0 - metrics.average_cost_ratio),
        "response_ratio": safe(metrics.average_response_ratio),
    }


def grade_task(state: GridState) -> float:
    score = clamp(state.metrics.task_score, SCORE_EPSILON, 1.0 - SCORE_EPSILON)
    return round(score, 4)


def grade_report(task_id: str, state: GridState) -> Dict[str, object]:
    return {
        "task_id": task_id,
        "score": grade_task(state),
        "breakdown": task_grade_breakdown(state),
    }


def grade_weekday_spike(state: GridState) -> float:
    return grade_task(state)


def grade_sunset_transition(state: GridState) -> float:
    return grade_task(state)


def grade_heatwave_failure(state: GridState) -> float:
    return grade_task(state)


def grade_storm_front_response(state: GridState) -> float:
    return grade_task(state)


def grade_winter_gas_shortage(state: GridState) -> float:
    return grade_task(state)


GRADERS: Dict[str, Callable[[GridState], float]] = {
    "weekday_spike": grade_weekday_spike,
    "sunset_transition": grade_sunset_transition,
    "heatwave_failure": grade_heatwave_failure,
    "storm_front_response": grade_storm_front_response,
    "winter_gas_shortage": grade_winter_gas_shortage,
}
