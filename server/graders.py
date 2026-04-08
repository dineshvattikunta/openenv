from __future__ import annotations

from env.models import GridState
from env.rewards import SCORE_EPSILON
from env.utils import clamp


def _score(state: GridState) -> float:
    return round(clamp(state.metrics.task_score, SCORE_EPSILON, 1.0 - SCORE_EPSILON), 4)


def grade_weekday_spike(state: GridState) -> float:
    return _score(state)


def grade_sunset_transition(state: GridState) -> float:
    return _score(state)


def grade_heatwave_failure(state: GridState) -> float:
    return _score(state)
