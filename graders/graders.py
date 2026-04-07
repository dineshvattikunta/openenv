from __future__ import annotations

from typing import Dict

from env.models import GridState
from env.rewards import SCORE_EPSILON
from env.utils import clamp


def task_grade_breakdown(state: GridState) -> Dict[str, float]:
    metrics = state.metrics
    return {
        "served_ratio": round(metrics.average_served_ratio, 4),
        "critical_served_ratio": round(metrics.average_critical_served_ratio, 4),
        "efficiency_ratio": round(metrics.average_efficiency_ratio, 4),
        "cost_efficiency_ratio": round(1.0 - metrics.average_cost_ratio, 4),
        "response_ratio": round(metrics.average_response_ratio, 4),
    }


def grade_task(state: GridState) -> Dict[str, object]:
    score = clamp(state.metrics.task_score, SCORE_EPSILON, 1.0 - SCORE_EPSILON)
    return {
        "task_id": state.scenario.task_name,
        "score": round(score, 4),
        "breakdown": task_grade_breakdown(state),
    }
