from __future__ import annotations

from .models import GridMetrics, GridStepInfo, TaskScenario
from .utils import clamp

SCORE_EPSILON = 0.01


def compute_step_reward(
    served_ratio: float,
    critical_served_ratio: float,
    efficiency_ratio: float,
    cost_score: float,
    response_ratio: float,
    penalty: float,
) -> float:
    reward = (
        0.40 * served_ratio
        + 0.25 * critical_served_ratio
        + 0.15 * efficiency_ratio
        + 0.10 * cost_score
        + 0.10 * response_ratio
        - penalty
    )
    return clamp(reward, -1.0, 1.0)


def compute_task_score(metrics: GridMetrics, scenario: TaskScenario) -> float:
    blackout_component = clamp(metrics.average_served_ratio, 0.0, 1.0)
    critical_component = clamp(metrics.average_critical_served_ratio, 0.0, 1.0)
    efficiency_component = clamp(metrics.average_efficiency_ratio, 0.0, 1.0)
    cost_component = clamp(1.0 - metrics.average_cost_ratio, 0.0, 1.0)
    response_component = clamp(metrics.average_response_ratio, 0.0, 1.0)

    score = (
        0.40 * blackout_component
        + 0.30 * critical_component
        + 0.15 * efficiency_component
        + 0.10 * cost_component
        + 0.05 * response_component
    )
    if scenario.task_name == "weekday_spike":
        score += 0.05 * response_component
    elif scenario.task_name == "sunset_transition":
        score += 0.05 * cost_component
    elif scenario.task_name == "heatwave_failure":
        score = (
            0.25 * blackout_component
            + 0.45 * critical_component
            + 0.10 * efficiency_component
            + 0.10 * cost_component
            + 0.10 * response_component
        )
        score -= 0.01 * metrics.critical_blackouts
        score -= 0.003 * metrics.blackouts
    score -= 0.02 * metrics.invalid_action_count
    # The validator requires task graders to stay strictly inside (0, 1),
    # never exactly 0.0 or 1.0.
    return clamp(score, SCORE_EPSILON, 1.0 - SCORE_EPSILON)


def summarize_reward(info: GridStepInfo) -> str:
    breakdown = info.reward_breakdown
    return (
        f"served={breakdown.blackout_score:.2f}, critical={breakdown.critical_service_score:.2f}, "
        f"eff={breakdown.efficiency_score:.2f}, cost={breakdown.cost_score:.2f}, "
        f"response={breakdown.response_score:.2f}, penalties={breakdown.penalties:.2f}"
    )
