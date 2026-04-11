from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

from baseline import heuristic_action
from env.engine import GridLoadBalancerEnv
from env.models import GridAction
from env.utils import action_to_log_string, extract_json_object

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
BENCHMARK = "power_grid"
SUCCESS_THRESHOLDS = {
    "weekday_spike": 0.75,
    "sunset_transition": 0.70,
    "heatwave_failure": 0.40,
    "storm_front_response": 0.62,
    "winter_gas_shortage": 0.45,
}
ALL_TASKS = [
    "weekday_spike",
    "sunset_transition",
    "heatwave_failure",
    "storm_front_response",
    "winter_gas_shortage",
]


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, round(float(value), 4)))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    clamped = _clamp(score)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={clamped:.4f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def build_prompt(observation) -> str:
    payload = {
        "task": observation.task_name,
        "step": observation.step,
        "max_steps": observation.max_steps,
        "summary": observation.summary,
        "reserve_margin_percent": observation.reserve_margin_percent,
        "weather": observation.weather,
        "active_faults": observation.active_faults,
        "critical_zones_at_risk": observation.critical_zones_at_risk,
        "zones": [zone.model_dump() for zone in observation.zones],
        "generators": [generator.model_dump() for generator in observation.generators],
        "batteries": [battery.model_dump() for battery in observation.batteries],
    }
    return (
        "You are controlling a city power grid. Return a single JSON object that matches the GridAction schema. "
        "Prioritize critical zones, avoid blackouts, then minimize cost. Return JSON only.\n"
        f"{json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}"
    )


def model_action(client: Optional[OpenAI], observation) -> GridAction:
    fallback = heuristic_action(observation)
    if client is None:
        return fallback
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a cautious grid dispatcher. Return valid JSON only."},
                {"role": "user", "content": build_prompt(observation)},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        content = completion.choices[0].message.content or ""
        data = extract_json_object(content)
        return GridAction.model_validate(data)
    except Exception:
        return fallback


def run_task(task_name: str) -> None:
    env = GridLoadBalancerEnv(task_name=task_name)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.5
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    try:
        observation = env.reset(task_name=task_name)
        done = False
        while not done:
            action = model_action(client, observation)
            result = env.step(action)
            steps_taken += 1
            rewards.append(result.reward)
            log_step(steps_taken, action_to_log_string(action), result.reward, result.done, result.info.last_action_error)
            observation = result.observation
            done = result.done
        raw_score = env.state().metrics.task_score
        score = _clamp(raw_score)
        success = raw_score >= SUCCESS_THRESHOLDS.get(task_name, 0.7)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    tasks_env = os.getenv("POWER_GRID_TASKS", "")
    if tasks_env:
        tasks = [t.strip() for t in tasks_env.split(",") if t.strip()]
    else:
        tasks = ALL_TASKS

    for task_name in tasks:
        run_task(task_name)


if __name__ == "__main__":
    main()
