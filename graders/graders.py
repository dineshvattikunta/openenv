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


def grade_task(state: GridState) -> Dict[str, object]:
    score = clamp(state.metrics.task_score, SCORE_EPSILON, 1.0 - SCORE_EPSILON)
    return {
        "task_id": state.scenario.task_name,
        "score": round(score, 4),
        "breakdown": task_grade_breakdown(state),
    }


def grade_weekday_spike(state: GridState) -> Dict[str, object]:
    grade = grade_task(state)
    grade["task_id"] = "weekday_spike"
    return grade


def grade_sunset_transition(state: GridState) -> Dict[str, object]:
    grade = grade_task(state)
    grade["task_id"] = "sunset_transition"
    return grade


def grade_heatwave_failure(state: GridState) -> Dict[str, object]:
    grade = grade_task(state)
    grade["task_id"] = "heatwave_failure"
    return grade


def grade_storm_front_response(state: GridState) -> Dict[str, object]:
    grade = grade_task(state)
    grade["task_id"] = "storm_front_response"
    return grade


def grade_winter_gas_shortage(state: GridState) -> Dict[str, object]:
    grade = grade_task(state)
    grade["task_id"] = "winter_gas_shortage"
    return grade


GRADERS: Dict[str, Callable[[GridState], Dict[str, object]]] = {
    "weekday_spike": grade_weekday_spike,
    "sunset_transition": grade_sunset_transition,
    "heatwave_failure": grade_heatwave_failure,
    "storm_front_response": grade_storm_front_response,
    "winter_gas_shortage": grade_winter_gas_shortage,
}
