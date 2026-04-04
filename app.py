from __future__ import annotations

import json
import os
from typing import Dict, List

import gradio as gr
from fastapi import FastAPI

from baseline import heuristic_action, run_baseline
from env.engine import GridLoadBalancerEnv
from env.models import GridAction
from env.rewards import summarize_reward
from env.tasks import list_tasks
from env.utils import action_to_log_string

CUSTOM_CSS = """
:root {
  --bg: #09121a;
  --panel: rgba(15, 30, 40, 0.92);
  --line: rgba(102, 221, 188, 0.16);
  --ink: #e7f4ef;
  --muted: #9ab8ae;
}
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(23, 92, 79, 0.35), transparent 42%),
    radial-gradient(circle at top right, rgba(23, 68, 111, 0.28), transparent 30%),
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
.metric-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.metric-card { padding: 14px 16px; border-radius: 16px; border: 1px solid var(--line); background: rgba(255,255,255,0.03); }
.metric-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.metric-value { font-size: 28px; font-weight: 700; }
.zone-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.zone-card { padding: 14px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.03); }
.zone-header { display:flex; justify-content:space-between; gap:12px; margin-bottom:10px; }
.pill { padding:4px 10px; border-radius:999px; font-size:12px; text-transform:uppercase; font-weight:700; }
.normal { background: rgba(49,196,141,0.18); color:#8bf0c7; }
.warning { background: rgba(246,196,83,0.18); color:#ffd977; }
.overload { background: rgba(240,91,91,0.18); color:#ffabab; }
.critical { background: rgba(255,80,80,0.22); color:#ffd2d2; }
.tiny { color: var(--muted); font-size: 13px; }
"""


def render_metrics(env: GridLoadBalancerEnv) -> str:
    state = env.state()
    critical = ", ".join(state.observation.critical_zones_at_risk) if state.observation.critical_zones_at_risk else "None"
    return f"""
    <div class="metric-strip">
      <div class="metric-card"><div class="metric-label">Task Score</div><div class="metric-value">{state.metrics.task_score:.2f}</div></div>
      <div class="metric-card"><div class="metric-label">Reserve Margin</div><div class="metric-value">{state.observation.reserve_margin_percent:.1f}%</div></div>
      <div class="metric-card"><div class="metric-label">Cumulative Reward</div><div class="metric-value">{state.metrics.cumulative_reward:.2f}</div></div>
      <div class="metric-card"><div class="metric-label">Critical Risk</div><div class="metric-value" style="font-size:18px">{critical}</div></div>
    </div>
    """


def render_zones(observation) -> str:
    rows = []
    for zone in observation.zones:
        rows.append(
            f"""
            <div class="zone-card">
              <div class="zone-header">
                <div>
                  <div><strong>{zone.label}</strong></div>
                  <div class="tiny">{zone.zone_id} | priority {zone.priority}{' | critical' if zone.critical else ''}</div>
                </div>
                <div class="pill {zone.status}">{zone.status}</div>
              </div>
              <div class="tiny">Demand {zone.demand_mw:.1f} MW | Served {zone.served_mw:.1f} MW | Unmet {zone.unmet_demand_mw:.1f} MW</div>
              <div class="tiny">Renewables {zone.renewable_output_mw:.1f} MW | Battery {zone.battery_soc_percent:.1f}% | Net transfer {zone.net_transfer_mw:.1f} MW</div>
              <div class="tiny">Load shed {zone.load_shed_mw:.1f} MW | Fault {'yes' if zone.fault_active else 'no'}</div>
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
        raw_json = gr.Code(value="", language="json", label="Observation Snapshot")
    with gr.Accordion("Task Metadata", open=False):
        gr.JSON(value=list_tasks())
    with gr.Accordion("Baseline Results", open=False):
        gr.Dataframe(headers=["Task", "Task Score", "Cumulative Reward", "Steps"], value=baseline_rows, interactive=False)

    start_button.click(init_scenario, inputs=task_selector, outputs=[env_state, log_state, metrics_html, zones_html, raw_json, action_table])
    step_button.click(run_one_step, inputs=[env_state, log_state], outputs=[env_state, log_state, metrics_html, zones_html, raw_json, action_table])
    auto_button.click(run_full, inputs=task_selector, outputs=[env_state, log_state, metrics_html, zones_html, raw_json, action_table])

app = FastAPI()


@app.post("/reset")
def reset_healthcheck():
    env = GridLoadBalancerEnv(task_name="weekday_spike")
    observation = env.reset(task_name="weekday_spike")
    env.close()
    return {
        "ok": True,
        "task": observation.task_name,
        "benchmark": observation.benchmark,
        "step": observation.step,
    }


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
