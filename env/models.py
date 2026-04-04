from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PowerTransfer(BaseModel):
    source_zone_id: str
    target_zone_id: str
    mw: float = Field(ge=0.0, default=0.0)


class GeneratorCommand(BaseModel):
    generator_id: str
    enabled: Optional[bool] = None
    target_output_mw: Optional[float] = Field(default=None, ge=0.0)


class BatteryCommand(BaseModel):
    battery_id: str
    mode: Literal["idle", "charge", "discharge"] = "idle"
    power_mw: float = Field(default=0.0, ge=0.0)


class LoadShedCommand(BaseModel):
    zone_id: str
    mw: float = Field(ge=0.0, default=0.0)


class GridAction(BaseModel):
    power_transfers: List[PowerTransfer] = Field(default_factory=list)
    generator_commands: List[GeneratorCommand] = Field(default_factory=list)
    battery_commands: List[BatteryCommand] = Field(default_factory=list)
    load_shed_commands: List[LoadShedCommand] = Field(default_factory=list)
    neighbor_import_mw: float = Field(default=0.0, ge=0.0)
    reason: str = ""


class ZoneConfig(BaseModel):
    zone_id: str
    label: str
    priority: int = Field(ge=1, le=5)
    critical: bool = False
    base_demand_mw: float = Field(gt=0.0)
    renewable_capacity_mw: float = Field(ge=0.0, default=0.0)


class GeneratorConfig(BaseModel):
    generator_id: str
    zone_id: str
    label: str
    min_output_mw: float = Field(ge=0.0, default=0.0)
    max_output_mw: float = Field(gt=0.0)
    ramp_mw_per_step: float = Field(gt=0.0)
    cost_per_mwh: float = Field(ge=0.0)
    startup_cost: float = Field(ge=0.0, default=0.0)
    initially_enabled: bool = False
    initial_output_mw: float = Field(ge=0.0, default=0.0)


class BatteryConfig(BaseModel):
    battery_id: str
    zone_id: str
    label: str
    energy_capacity_mwh: float = Field(gt=0.0)
    max_charge_mw: float = Field(gt=0.0)
    max_discharge_mw: float = Field(gt=0.0)
    round_trip_efficiency: float = Field(gt=0.0, le=1.0, default=0.94)
    initial_soc_percent: float = Field(ge=0.0, le=100.0, default=50.0)


class TaskEvent(BaseModel):
    step: int = Field(ge=1)
    event_type: Literal["generator_failure", "fault", "line_derate", "demand_shock"]
    target_id: str
    value: float = 0.0
    duration_steps: int = Field(ge=1, default=1)
    note: str = ""


class TaskScenario(BaseModel):
    task_name: str
    benchmark: str = "power_grid"
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int = Field(gt=0)
    time_step_minutes: int = Field(gt=0, default=5)
    line_capacity_mw: float = Field(gt=0.0, default=35.0)
    neighbor_import_capacity_mw: float = Field(ge=0.0, default=0.0)
    neighbor_import_cost_per_mwh: float = Field(ge=0.0, default=85.0)
    objective: str
    zones: List[ZoneConfig]
    generators: List[GeneratorConfig]
    batteries: List[BatteryConfig]
    demand_profile: Dict[str, List[float]]
    renewable_profile: Dict[str, List[float]]
    events: List[TaskEvent] = Field(default_factory=list)


class ZoneObservation(BaseModel):
    zone_id: str
    label: str
    priority: int
    critical: bool
    demand_mw: float
    served_mw: float
    unmet_demand_mw: float
    local_generation_mw: float
    renewable_output_mw: float
    battery_soc_percent: float
    net_transfer_mw: float
    load_shed_mw: float
    fault_active: bool
    status: Literal["normal", "warning", "overload", "critical"]


class GeneratorObservation(BaseModel):
    generator_id: str
    zone_id: str
    label: str
    enabled: bool
    available: bool
    output_mw: float
    max_output_mw: float
    cost_per_mwh: float


class BatteryObservation(BaseModel):
    battery_id: str
    zone_id: str
    label: str
    soc_percent: float
    mode: Literal["idle", "charge", "discharge"]
    power_mw: float
    available_charge_mw: float
    available_discharge_mw: float


class RewardBreakdown(BaseModel):
    blackout_score: float
    critical_service_score: float
    efficiency_score: float
    cost_score: float
    response_score: float
    penalties: float
    total_reward: float


class GridStepInfo(BaseModel):
    valid_action: bool
    last_action_error: Optional[str] = None
    transfer_clipped_mw: float = 0.0
    import_used_mw: float = 0.0
    total_operating_cost: float = 0.0
    total_wasted_energy_mw: float = 0.0
    critical_zones_at_risk: List[str] = Field(default_factory=list)
    overloaded_lines: int = 0
    reward_breakdown: RewardBreakdown


class GridObservation(BaseModel):
    task_name: str
    benchmark: str
    step: int
    max_steps: int
    time_of_day: str
    weather: str
    reserve_margin_percent: float
    active_faults: List[str] = Field(default_factory=list)
    overloaded_lines: int = 0
    critical_zones_at_risk: List[str] = Field(default_factory=list)
    last_action_error: Optional[str] = None
    zones: List[ZoneObservation]
    generators: List[GeneratorObservation]
    batteries: List[BatteryObservation]
    summary: str


class GridMetrics(BaseModel):
    cumulative_reward: float = 0.0
    reward_history: List[float] = Field(default_factory=list)
    average_served_ratio: float = 0.0
    average_critical_served_ratio: float = 0.0
    average_efficiency_ratio: float = 0.0
    average_cost_ratio: float = 0.0
    average_response_ratio: float = 0.0
    total_unmet_mwh: float = 0.0
    total_critical_unmet_mwh: float = 0.0
    total_cost: float = 0.0
    invalid_action_count: int = 0
    blackouts: int = 0
    critical_blackouts: int = 0
    task_score: float = 0.0


class GridState(BaseModel):
    scenario: TaskScenario
    observation: GridObservation
    metrics: GridMetrics
    generator_outputs_mw: Dict[str, float]
    generator_enabled: Dict[str, bool]
    battery_soc_percent: Dict[str, float]
    battery_mode: Dict[str, str]
    last_info: GridStepInfo


class GridStepResult(BaseModel):
    observation: GridObservation
    reward: float
    done: bool
    info: GridStepInfo
