---
title: Grid Load Balancer
emoji: "⚡"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---
# Grid Load Balancer

Grid Load Balancer is a real-world OpenEnv-style environment where an AI agent acts as a city-scale power dispatcher. The agent must keep electricity flowing across multiple zones, react to demand spikes, compensate for renewable drop-offs, activate backup generation, manage batteries, and protect critical infrastructure during failures.

This repository includes:

- a deterministic simulator
- 5 graded tasks from easy to hard
- typed Pydantic models for action, observation, and state
- a root-level `inference.py` using the OpenAI client
- a baseline heuristic
- a Gradio Space UI
- Docker packaging

## Why this benchmark is strong

This environment models a real control problem that utilities and virtual power plant operators face today:

- balancing load between stressed zones
- using batteries to smooth renewable volatility
- dispatching expensive backup generation only when needed
- protecting hospitals, airports, water systems, and data centers during failures

It is operational, explainable, and easy to judge.

## Tasks

### `weekday_spike`

- Difficulty: easy
- Scenario: 3 zones, weekday business demand surge
- Objective: reroute power and restore balance quickly

### `sunset_transition`

- Difficulty: medium
- Scenario: 8 zones, evening peak, solar fading, variable wind
- Objective: keep service continuous while using batteries and backup efficiently

### `heatwave_failure`

- Difficulty: hard
- Scenario: 15 zones, heatwave demand, main generator failure, critical infrastructure under risk
- Objective: protect critical zones and avoid cascading outages

### `storm_front_response`

- Difficulty: medium
- Scenario: 8 zones, severe storm front, repeated line derates and feeder faults
- Objective: keep critical services online while transfer capacity is unstable

### `winter_gas_shortage`

- Difficulty: hard
- Scenario: 15 zones, winter peak demand, intermittent gas generation failures
- Objective: preserve critical uptime when supply-side outages and import limits overlap

## Observation space

The agent sees:

- task name, step, weather, and time of day
- reserve margin and active faults
- overloaded lines and critical zones at risk
- zone-level demand, served load, unmet load, transfers, renewables, load shed, and battery state
- generator availability, output, and cost
- battery charge and discharge capacity

## Action space

The `GridAction` model supports:

- `power_transfers`
- `generator_commands`
- `battery_commands`
- `load_shed_commands`
- `neighbor_import_mw`
- `reason`

Invalid actions are clipped and recorded through `last_action_error` instead of crashing the episode.

## Reward design

Per-step reward is shaped as:

`0.40 * blackout_score + 0.25 * critical_service_score + 0.15 * efficiency_score + 0.10 * cost_score + 0.10 * response_score - penalties`

The grader score is normalized to `[0.0, 1.0]` and is deterministic.

## Project layout

```text
.
|- env/
|  |- __init__.py
|  |- engine.py
|  |- models.py
|  |- rewards.py
|  |- tasks.py
|  `- utils.py
|- scripts/
|  `- validate-submission.sh
|- app.py
|- baseline.py
|- inference.py
|- openenv.yaml
|- Dockerfile
|- requirements.txt
`- README.md
```

## Inference script

The required root-level `inference.py`:

- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- uses the OpenAI client for all LLM calls
- emits exactly `[START]`, `[STEP]`, and `[END]` lines
- falls back to a deterministic heuristic if model inference fails

Example:

```powershell
$env:HF_TOKEN="your-token"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

## Local setup

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run baseline evaluation:

```powershell
python baseline.py
```

Run the demo app:

```powershell
python app.py
```

Run the baseline inference script:

```powershell
python inference.py
```

## Hugging Face Space

The project ships with a Docker-based Space setup:

- SDK: Docker
- Entry: `python app.py`
- Port: `7860`
- Healthcheck route: `POST /reset`

Recommended environment variables:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`

## Validation

Before submission, verify:

- `openenv.yaml` exists
- `inference.py` is in the repo root
- the Docker image builds
- the Space launches
- `POST /reset` returns `200`
- task scores vary meaningfully across scenarios
- `openenv validate` passes in your target environment

Helper script:

```bash
./scripts/validate-submission.sh https://your-space.hf.space .
```
