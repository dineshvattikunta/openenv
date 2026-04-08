from __future__ import annotations

import json
import os
from typing import Dict, List

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from baseline import heuristic_action, run_baseline
from env.engine import GridLoadBalancerEnv
from env.models import GridAction
from env.rewards import summarize_reward
from env.tasks import list_tasks
from env.utils import action_to_log_string
from graders.graders import GRADERS, task_grade_breakdown

CUSTOM_CSS = """
:root {
  --bg: #08131d;
  --panel: rgba(16, 26, 36, 0.96);
  --panel-soft: rgba(24, 38, 51, 0.96);
  --line: rgba(120, 196, 171, 0.22);
  --ink: #eef7f3;
  --muted: #9fb7b0;
  --good: #38c793;
  --warn: #f0b44c;
  --bad: #ef6b6b;
  --accent: #60d3ff;
}
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(24, 96, 81, 0.34), transparent 42%),
    radial-gradient(circle at top right, rgba(18, 82, 121, 0.24), transparent 30%),
    linear-gradient(180deg, #081018 0%, #0d1820 100%);
  color: var(--ink);
  font-family: "Segoe UI", "Trebuchet MS", sans-serif;
}
.hero, .panel {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: var(--panel);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
}
.hero { padding: 20px 24px; margin-bottom: 10px; }
.metric-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
.metric-card { padding: 16px 18px; border-radius: 16px; border: 1px solid var(--line); background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); }
.metric-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
.status-line { padding: 14px 16px; border-radius: 16px; background: var(--panel-soft); border: 1px solid var(--line); margin-bottom: 14px; }
.status-title { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
.status-copy { color: var(--muted); line-height: 1.5; }
.zone-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.zone-card { padding: 16px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); }
.zone-header { display:flex; justify-content:space-between; align-items: center; gap:12px; margin-bottom:12px; }
.pill { padding:4px 10px; border-radius:999px; font-size:12px; text-transform:uppercase; font-weight:700; }
.normal { background: rgba(56,199,147,0.18); color:#8ef3c9; }
.warning { background: rgba(240,180,76,0.18); color:#ffd47a; }
.overload { background: rgba(239,107,107,0.18); color:#ffb0b0; }
.critical { background: rgba(239,107,107,0.24); color:#ffe0e0; }
.tiny { color: var(--muted); font-size: 13px; }
.zone-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.zone-table td { padding: 6px 0; vertical-align: top; }
.zone-table td:first-child { color: var(--muted); width: 48%; }
.zone-flag { margin-top: 10px; padding: 8px 10px; border-radius: 12px; font-size: 13px; }
.flag-good { background: rgba(56,199,147,0.12); color: #8ef3c9; }
.flag-warn { background: rgba(240,180,76,0.12); color: #ffd47a; }
.flag-bad { background: rgba(239,107,107,0.14); color: #ffb0b0; }
"""

SUCCESS_THRESHOLDS = {
    "weekday_spike": 0.75,
    "sunset_transition": 0.70,
    "heatwave_failure": 0.40,
    "storm_front_response": 0.62,
    "winter_gas_shortage": 0.45,
}

TASK_GRADERS = {
    "weekday_spike": "grader.grade_weekday_spike",
    "sunset_transition": "grader.grade_sunset_transition",
    "heatwave_failure": "grader.grade_heatwave_failure",
    "storm_front_response": "grader.grade_storm_front_response",
    "winter_gas_shortage": "grader.grade_winter_gas_shortage",
}

API_ENV = GridLoadBalancerEnv(task_name="weekday_spike")
API_ENV.reset(task_name="weekday_spike")


class ResetRequest(BaseModel):
    task_id: str = "weekday_spike"


class StepRequest(BaseModel):
    action: dict


def render_metrics(env: GridLoadBalancerEnv) -> str:
    state = env.state()
    critical = ", ".join(state.observation.critical_zones_at_risk) if state.observation.critical_zones_at_risk else "None"
    score = state.metrics.task_score
    if score >= 0.8:
        verdict = "Strong run"
    elif score >= 0.6:
        verdict = "Stable run"
    elif score >= 0.4:
        verdict = "Needs improvement"
    else:
        verdict = "Weak run"

    if state.observation.critical_zones_at_risk:
        summary = f"Critical infrastructure is currently at risk in: {critical}."
    elif state.observation.reserve_margin_percent < 15:
        summary = "The grid is running, but spare capacity is low. One new fault could create outages."
    else:
        summary = "The grid is stable right now. No critical zones are currently in danger."

    return f"""
    <div class="metric-strip">
      <div class="metric-card"><div class="metric-label">Task Score</div><div class="metric-value">{state.metrics.task_score:.2f}</div></div>
      <div class="metric-card"><div class="metric-label">Reserve Margin</div><div class="metric-value">{state.observation.reserve_margin_percent:.1f}%</div></div>
      <div class="metric-card"><div class="metric-label">Cumulative Reward</div><div class="metric-value">{state.metrics.cumulative_reward:.2f}</div></div>
      <div class="metric-card"><div class="metric-label">Critical Risk</div><div class="metric-value" style="font-size:18px">{critical}</div></div>
    </div>
    <div class="status-line">
      <div class="status-title">What This Means: {verdict}</div>
      <div class="status-copy">{summary}</div>
    </div>
    """


def render_zones(observation) -> str:
    rows = []
    for zone in observation.zones:
        if zone.unmet_demand_mw <= 0.01:
            flag_class = "flag-good"
            flag_text = "This zone is fully served."
        elif zone.critical:
            flag_class = "flag-bad"
            flag_text = "This is a critical zone and it is short on power."
        elif zone.unmet_demand_mw > zone.demand_mw * 0.25:
            flag_class = "flag-bad"
            flag_text = "This zone is heavily under-supplied."
        else:
            flag_class = "flag-warn"
            flag_text = "This zone is running, but some demand is not being met."
        rows.append(
            f"""
            <div class="zone-card">
              <div class="zone-header">
                <div>
                  <div><strong>{zone.label}</strong></div>
                  <div class="tiny">Zone ID: {zone.zone_id} | Priority: {zone.priority}{' | Critical zone' if zone.critical else ''}</div>
                </div>
                <div class="pill {zone.status}">{zone.status}</div>
              </div>
              <table class="zone-table">
                <tr><td>Power Needed</td><td>{zone.demand_mw:.1f} MW</td></tr>
                <tr><td>Power Delivered</td><td>{zone.served_mw:.1f} MW</td></tr>
                <tr><td>Missing Power</td><td>{zone.unmet_demand_mw:.1f} MW</td></tr>
                <tr><td>Renewable Power</td><td>{zone.renewable_output_mw:.1f} MW</td></tr>
                <tr><td>Battery Level</td><td>{zone.battery_soc_percent:.1f}%</td></tr>
                <tr><td>Net Power Transfer</td><td>{zone.net_transfer_mw:.1f} MW</td></tr>
                <tr><td>Load Shed</td><td>{zone.load_shed_mw:.1f} MW</td></tr>
                <tr><td>Fault Present</td><td>{"Yes" if zone.fault_active else "No"}</td></tr>
              </table>
              <div class="zone-flag {flag_class}">{flag_text}</div>
            </div>
            """
        )
    return f'<div class="zone-grid">{"".join(rows)}</div>'


def logs_to_rows(logs: List[Dict[str, str]]) -> List[List[str]]:
    return [[item["step"], item["action"], item["reward"], item["error"], item["summary"]] for item in logs]


def init_scenario(task_name: str):
    env = GridLoadBalancerEnv(task_name=task_name)
    observation = env.reset(task_name=task_name)
    logs: List[Dict[str, str]] = []
    return env, logs, render_metrics(env), render_zones(observation), json.dumps(observation.model_dump(), indent=2), logs_to_rows(logs)


def run_one_step(env: GridLoadBalancerEnv, logs: List[Dict[str, str]]):
    observation = env.state().observation
    action: GridAction = heuristic_action(observation)
    result = env.step(action)
    logs = list(logs)
    logs.append({"step": str(result.observation.step), "action": action_to_log_string(action), "reward": f"{result.reward:.2f}", "error": result.info.last_action_error or "null", "summary": summarize_reward(result.info)})
    return env, logs, render_metrics(env), render_zones(result.observation), json.dumps(result.observation.model_dump(), indent=2), logs_to_rows(logs)


def run_full(task_name: str):
    env = GridLoadBalancerEnv(task_name=task_name)
    observation = env.reset(task_name=task_name)
    logs: List[Dict[str, str]] = []
    done = False
    while not done:
        action = heuristic_action(observation)
        result = env.step(action)
        logs.append({"step": str(result.observation.step), "action": action_to_log_string(action), "reward": f"{result.reward:.2f}", "error": result.info.last_action_error or "null", "summary": summarize_reward(result.info)})
        observation = result.observation
        done = result.done
    return env, logs, render_metrics(env), render_zones(observation), json.dumps(observation.model_dump(), indent=2), logs_to_rows(logs)


baseline_rows = [[task, f"{res['task_score']:.4f}", f"{res['cumulative_reward']:.4f}", str(res["steps"])] for task, res in {name: run_baseline(name) for name in list_tasks()}.items()]

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    env_state = gr.State()
    log_state = gr.State([])
    gr.HTML(
        """
        <div class="hero">
          <div style="font-size:13px; text-transform:uppercase; letter-spacing:0.12em; color:#9ab8ae;">OpenEnv Submission</div>
          <h1 style="margin:8px 0 10px 0;">Grid Load Balancer</h1>
          <p style="margin:0; max-width:880px; color:#d7ebe3;">
            A real-time AI control room for urban electricity distribution. The agent reroutes power,
            dispatches backup generation, uses batteries, and protects critical infrastructure during
            spikes, renewable drop-offs, and generator failures.
          </p>
        </div>
        """
    )
    with gr.Row():
        task_selector = gr.Dropdown(choices=list(list_tasks().keys()), value="weekday_spike", label="Scenario")
        start_button = gr.Button("Initialize Scenario", variant="primary")
        step_button = gr.Button("Run One Baseline Step")
        auto_button = gr.Button("Run Full Baseline")
    metrics_html = gr.HTML(value="", elem_classes=["panel"])
    zones_html = gr.HTML(value="", elem_classes=["panel"])
    with gr.Row():
        action_table = gr.Dataframe(headers=["Step", "Action", "Reward", "Error", "Reward Summary"], value=[], interactive=False, wrap=True, label="Action Log")
        raw_json = gr.Code(value="", language="json", label="Raw Observation JSON")
    with gr.Accordion("Task Metadata", open=False):
        gr.JSON(value=list_tasks())
    with gr.Accordion("Baseline Results", open=False):
        gr.Dataframe(headers=["Task", "Task Score", "Cumulative Reward", "Steps"], value=baseline_rows, interactive=False)

    start_button.click(init_scenario, inputs=task_selector, outputs=[env_state, log_state, metrics_html, zones_html, raw_json, action_table])
    step_button.click(run_one_step, inputs=[env_state, log_state], outputs=[env_state, log_state, metrics_html, zones_html, raw_json, action_table])
    auto_button.click(run_full, inputs=task_selector, outputs=[env_state, log_state, metrics_html, zones_html, raw_json, action_table])

app = FastAPI()

def _reset_payload():
    observation = API_ENV.reset(task_name="weekday_spike")
    return {
        "ok": True,
        "task": observation.task_name,
        "benchmark": observation.benchmark,
        "step": observation.step,
    }


@app.get("/reset")
def reset_healthcheck_get():
    return _reset_payload()


@app.post("/reset")
def reset_healthcheck_post(payload: ResetRequest | None = None):
    task_id = payload.task_id if payload else "weekday_spike"
    observation = API_ENV.reset(task_name=task_id)
    return {
        "ok": True,
        "task": observation.task_name,
        "benchmark": observation.benchmark,
        "step": observation.step,
    }


@app.post("/step")
def step_env(payload: StepRequest):
    action = GridAction.model_validate(payload.action)
    result = API_ENV.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info.model_dump(),
    }


@app.get("/state")
def get_state():
    return API_ENV.state().model_dump()


@app.get("/tasks")
def get_tasks():
    tasks = []
    for task_id, meta in list_tasks().items():
        tasks.append(
            {
                "id": task_id,
                "name": task_id.replace("_", " ").title(),
                "difficulty": meta["difficulty"],
                "description": meta["objective"],
                "max_steps": {
                    "weekday_spike": 8,
                    "sunset_transition": 10,
                    "heatwave_failure": 12,
                    "storm_front_response": 11,
                    "winter_gas_shortage": 13,
                }.get(task_id, 10),
                "success_threshold": SUCCESS_THRESHOLDS.get(task_id, 0.5),
                "grader": task_id in GRADERS,
                "grader_path": TASK_GRADERS.get(task_id, "grader.grade_weekday_spike"),
            }
        )
    return {"tasks": tasks}


@app.get("/validate")
def validate_env():
    tasks = list_tasks()
    return {
        "valid": True,
        "env_name": "grid-load-balancer",
        "version": "0.1.0",
        "task_count": len(tasks),
        "graders_enabled": True,
        "tasks": list(tasks.keys()),
    }


@app.get("/grade/{task_id}")
def grade_env(task_id: str):
    if task_id not in GRADERS:
        return {"task_id": task_id, "score": 0.01, "breakdown": {"error_score": 0.01}}
    env = GridLoadBalancerEnv(task_name=task_id)
    observation = env.reset(task_name=task_id)
    done = False
    while not done:
        action = heuristic_action(observation)
        result = env.step(action)
        observation = result.observation
        done = result.done
    state = env.state()
    score = GRADERS[task_id](state)
    grade = {
        "task_id": task_id,
        "score": round(float(score), 2),
        "breakdown": task_grade_breakdown(state),
    }
    env.close()
    return grade


@app.get("/grader/{task_id}")
def grader_env_by_task(task_id: str):
    # Alias endpoint for validators that expect /grader/{task_id}.
    return grade_env(task_id)


@app.get("/grader")
def grader_env(task_id: str | None = None):
    # Alias endpoint for validators that expect /grader or /grader?task_id=...
    if task_id is not None:
        return grade_env(task_id)

    scores = []
    for tid in list_tasks().keys():
        result = grade_env(tid)
        scores.append({"task_id": tid, "score": result.get("score", 0.001)})
    return {"scores": scores}


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
