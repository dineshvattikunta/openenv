from __future__ import annotations

from typing import Dict

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
