# Grid Load Balancer

## Core Pitch

Grid Load Balancer is a real-world OpenEnv environment where an AI agent acts as a city-scale power distribution controller. The agent must continuously balance electricity across zones, manage batteries and backup generators, react to renewable volatility, and protect critical infrastructure during failures. The environment is designed to feel operational rather than game-like: every action mirrors decisions made by utility control rooms, microgrid operators, and virtual power plant software.

This directly targets the judging criteria:

- Real-world utility: power balancing, outage prevention, and critical-load protection are high-value industry problems.
- Task and grader quality: three tasks form a clear easy-to-hard progression with deterministic scoring.
- Environment design: typed state/action models, bounded action space, interpretable reward shaping, and proper episode lifecycle.
- Code quality and spec compliance: clean module split for models, simulation, tasks, rewards, baseline, Space app, and containerization.
- Creativity: utility-grid orchestration is novel, explainable, visually compelling, and highly relevant to renewable energy operations.

## One-Line Story

"An AI grid operator that prevents blackouts by rerouting power, dispatching backup energy, and protecting hospitals, airports, and data centers in real time."

## Why This Idea Is Strong

- It is clearly not a toy.
- It has direct industrial relevance.
- It is easy for judges to understand quickly.
- It supports meaningful partial rewards instead of binary pass/fail.
- It gives room for smart agent behavior without requiring heavy physics simulation.
- It is highly demoable in a Hugging Face Space.

## Submission Requirements Mapped To Design

### Real-world task simulation

The environment simulates a grid operator's control loop:

- observe demand, supply, weather, battery state, faults, and reserve margin
- choose dispatch and routing actions
- see immediate operational consequences
- get rewarded for maintaining stable service efficiently and safely

### OpenEnv spec compliance

We will implement:

- typed Pydantic observation, action, and reward-related info models
- `reset()` returning initial observation
- `step(action)` returning observation, reward, done, and info
- `state()` returning full current environment state
- `openenv.yaml` with metadata, tasks, and interfaces

### Minimum 3 tasks with graders

We will include three deterministic tasks:

1. `weekday_spike`
2. `sunset_transition`
3. `heatwave_failure`

Each task has:

- fixed objective
- deterministic scenario generator from seed
- agent grader score in `[0.0, 1.0]`
- increasing complexity

### Meaningful reward function

Reward is shaped per step to reflect real operations:

- protect demand fulfillment
- preserve critical service
- minimize unnecessary shedding
- use cheaper cleaner sources first
- respond quickly to instability
- penalize unsafe oscillation and waste

### Baseline inference script

The baseline script will:

- be named `inference.py`
- live at repo root
- use the OpenAI client
- read `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- print exactly `[START]`, `[STEP]`, and `[END]` lines in the required format
- run reproducibly on all tasks

### HF Space + Docker

We will include:

- a working `Dockerfile`
- `app.py` Space UI with live dashboard
- environment startup compatible with validation flow

### Documentation

The README will include:

- environment motivation
- observation and action space
- task descriptions
- scoring logic
- setup and run instructions
- validation steps

## Environment Design

### Observation

The agent sees a compact operational snapshot:

- `time_step`
- `time_of_day`
- `weather`
- `total_generation_available_mw`
- `neighbor_import_capacity_mw`
- `zones`

Each zone contains:

- `zone_id`
- `demand_mw`
- `served_mw`
- `priority`
- `status`
- `local_storage_mwh`
- `net_import_export_mw`
- `fault_active`
- `renewable_output_mw`

Global system indicators:

- `active_generators`
- `available_generators`
- `battery_soc_percent`
- `reserve_margin_percent`
- `overloaded_lines`
- `critical_zones_at_risk`
- `last_action_error`

### Action

The action space should be structured, bounded, and judge-friendly.

Recommended action model:

- `power_transfers`: list of source-to-target transfers in MW
- `generator_commands`: list of generator activate/deactivate or dispatch delta commands
- `battery_commands`: list of charge/discharge commands by asset
- `load_shed_commands`: list of zones and MW to shed
- `neighbor_import_mw`: optional import request
- `reason`: short free-text rationale for observability in logs

Important constraint:

The environment must clip invalid values safely and record violations in `info` and `last_action_error` rather than crash on normal bad actions.

## System Dynamics

Each step simulates one control interval, for example 5 minutes.

Update order:

1. Apply exogenous changes for the task:
   demand variation, renewable change, fault events, generator failures.
2. Validate and apply agent action.
3. Recompute zone supply-demand balance.
4. Update storage and generator states.
5. Detect overload, unmet demand, blackout conditions, and critical failures.
6. Compute step reward.
7. Check termination conditions.

## Task Design

### Task 1: `weekday_spike`

Scenario:

- 3 zones
- one sudden commercial or residential demand spike
- no major faults
- limited but sufficient transfer capacity

Objective:

- restore balanced service quickly without load shedding

Judge focus:

- balance restoration speed
- zero blackout behavior
- avoiding overreaction

Why it works:

- very easy to understand
- teaches basic rerouting
- stable baseline task for validators and demos

### Task 2: `sunset_transition`

Scenario:

- 8 zones
- evening demand rise
- solar output drops over time
- wind output fluctuates
- batteries and backup generators matter

Objective:

- maintain uninterrupted service while minimizing expensive backup overuse

Judge focus:

- zero blackouts
- cost-aware dispatch
- smart battery timing

Why it works:

- introduces realistic renewable intermittency
- rewards planning, not just reaction

### Task 3: `heatwave_failure`

Scenario:

- 15 zones
- extreme demand from cooling
- one main generator fails suddenly
- 3 critical zones: hospital, airport, data center
- non-critical shedding may be required

Objective:

- preserve critical-zone uptime and system stability under crisis

Judge focus:

- critical-zone protection
- intelligent sacrifice of low-priority loads
- resilience under cascading risk

Why it works:

- operationally dramatic
- strongly real-world
- high ceiling for strong agents

## Reward Design

Step reward should stay interpretable and bounded.

Recommended weighted score:

`reward = 0.40 * blackout_score + 0.25 * critical_service_score + 0.15 * efficiency_score + 0.10 * cost_score + 0.10 * response_score - penalties`

Where:

- `blackout_score`: fraction of demand served across all zones
- `critical_service_score`: fraction of critical demand served, heavily weighted
- `efficiency_score`: reward for low waste, reduced curtailment, and smooth routing
- `cost_score`: reward for using renewables and batteries before expensive backup imports
- `response_score`: reward for reducing overloads and restoring reserve margin quickly

Penalties:

- invalid action penalty
- oscillation penalty for rapid toggling of the same generator or battery behavior
- excessive non-critical shedding penalty
- hard penalty if any critical zone blacks out

Important:

The grader score and the step reward should be related but not identical. Step reward guides learning; grader score evaluates task success cleanly.

## Grader Strategy

Each task should expose a final normalized score in `[0.0, 1.0]`.

### Suggested grader outputs

`weekday_spike`

- 1.0 if no blackout and balance restored within target horizon
- partial credit based on unmet demand and recovery time

`sunset_transition`

- combines service continuity, battery usage efficiency, and backup cost efficiency
- zero service interruption should dominate

`heatwave_failure`

- critical-zone uptime is dominant
- non-critical service quality gives secondary credit
- unnecessary widespread shedding reduces score sharply

This protects us from the "all scores look the same" disqualification risk.

## Determinism and Validation Safety

To score well and pass validation:

- all tasks must be reproducible from seed
- graders must be deterministic
- invalid actions should not crash the episode
- score outputs must stay in `[0.0, 1.0]`
- episodes should finish well under runtime limits

## Recommended File Layout

```text
power-grid-env/
├── env/
│   ├── __init__.py
│   ├── engine.py          # simulation loop
│   ├── tasks.py           # scenario registry and generators
│   ├── rewards.py         # step reward + task graders
│   ├── models.py          # typed observation/action/state/info models
│   └── utils.py           # clipping, normalization, helpers
├── inference.py           # required baseline inference script
├── baseline.py            # optional local scripted baseline
├── app.py                 # HF Space dashboard
├── openenv.yaml           # OpenEnv metadata/spec
├── Dockerfile
├── requirements.txt
└── README.md
```

## Baseline Agent Strategy

The baseline should be simple, reproducible, and non-trivial.

Prompt the model with:

- current zone loads
- critical zones
- available generators
- battery state
- weather and renewable trend
- last action result

Expected baseline behavior:

- prioritize critical zones first
- reroute surplus before importing
- use batteries to absorb short spikes
- activate backup generators only when projected deficit persists
- shed low-priority load only as last resort

Important compliance note:

The required stdout format must be followed exactly.

The sample currently includes a `score=` field in `[END]`, but the text requirement says the line must be:

`[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

For strict compliance, we should not include extra fields in `[END]` unless the validator explicitly expects them. Safer choice: omit `score=` from the final line.

## HF Space Demo Concept

The demo should make the project feel premium and immediately understandable.

### Main panels

- city grid overview map
- zone cards with demand, served power, storage, and risk level
- generator and battery dispatch panel
- live reward and stability indicators
- action log timeline

### Visual language

- green: stable
- amber: stressed
- red: overloaded or outage risk
- blue: battery or import flows

### Demo modes

- run baseline agent
- replay a saved scenario
- compare before-vs-after stabilization

## What Will Impress Judges Most

- protecting hospitals, airports, and data centers in the hard task
- intelligent partial load shedding instead of blunt blackout behavior
- clean reward decomposition
- a polished dashboard showing why an action helped
- strong documentation that explains "why this matters in the real world"

## Risks To Avoid

- turning it into a game instead of an operational simulator
- overly complex physics that hurt reliability
- action space that is too large or hard for an LLM baseline to use
- graders that are vague or collapse to similar scores
- a baseline script that prints the wrong stdout format
- a Docker image that is too heavy or slow for validator limits

## Implementation Order

1. Define typed models for zones, assets, observations, actions, and step info.
2. Build the deterministic simulation engine with simple but realistic constraints.
3. Add reward shaping and final graders.
4. Register the three tasks and seed-based scenario generation.
5. Add `openenv.yaml`.
6. Build the required `inference.py` with exact logging format.
7. Build the Space UI in `app.py`.
8. Add Dockerfile, requirements, and README.
9. Run validation and tighten any spec mismatches.

## Strong Project Tagline Options

- `Grid Load Balancer: AI that prevents city-scale blackouts`
- `PowerFlow Guardian: an AI control room for modern electric grids`
- `Critical Grid Ops: balancing energy, backups, and resilience in real time`

## Recommended Final Positioning

The strongest framing is:

"A realistic AI grid-operations environment where agents learn to keep cities powered during spikes, renewable drop-offs, and infrastructure failures while protecting critical services and minimizing cost."

That framing hits utility, realism, task clarity, and novelty in one sentence.
