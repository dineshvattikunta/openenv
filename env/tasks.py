from __future__ import annotations

from typing import Dict, List

from .models import BatteryConfig, GeneratorConfig, TaskEvent, TaskScenario, ZoneConfig


WEATHER_BY_TASK = {
    "weekday_spike": "clear",
    "sunset_transition": "sunset_windy",
    "heatwave_failure": "heatwave",
}


TIME_OF_DAY_BY_TASK = {
    "weekday_spike": "14:00",
    "sunset_transition": "18:30",
    "heatwave_failure": "15:45",
}


def _repeat(values: List[float], target: int) -> List[float]:
    return values[:target] if len(values) >= target else values + [values[-1]] * (target - len(values))


def build_weekday_spike() -> TaskScenario:
    max_steps = 8
    zones = [
        ZoneConfig(zone_id="north", label="North Residential", priority=2, base_demand_mw=48, renewable_capacity_mw=18),
        ZoneConfig(zone_id="central", label="Central Business", priority=3, base_demand_mw=62, renewable_capacity_mw=8),
        ZoneConfig(zone_id="south", label="South Industrial", priority=2, base_demand_mw=55, renewable_capacity_mw=12),
    ]
    generators = [
        GeneratorConfig(
            generator_id="gas_n1",
            zone_id="north",
            label="North Gas Peaker",
            min_output_mw=10,
            max_output_mw=60,
            ramp_mw_per_step=18,
            cost_per_mwh=58,
            initially_enabled=True,
            initial_output_mw=26,
        ),
        GeneratorConfig(
            generator_id="gas_s1",
            zone_id="south",
            label="South Gas Turbine",
            min_output_mw=8,
            max_output_mw=54,
            ramp_mw_per_step=18,
            cost_per_mwh=54,
            initially_enabled=True,
            initial_output_mw=24,
        ),
    ]
    batteries = [
        BatteryConfig(
            battery_id="bat_c1",
            zone_id="central",
            label="Central Battery",
            energy_capacity_mwh=60,
            max_charge_mw=24,
            max_discharge_mw=24,
            initial_soc_percent=62,
        )
    ]
    demand_profile = {
        "north": _repeat([1.00, 1.02, 1.02, 0.99, 0.98, 0.97, 0.96, 0.95], max_steps),
        "central": _repeat([1.00, 1.25, 1.30, 1.15, 1.05, 1.00, 0.98, 0.96], max_steps),
        "south": _repeat([1.00, 1.01, 1.02, 1.00, 1.00, 0.99, 0.98, 0.98], max_steps),
    }
    renewable_profile = {
        "north": _repeat([0.75, 0.78, 0.76, 0.74, 0.72, 0.70, 0.69, 0.68], max_steps),
        "central": _repeat([0.60, 0.58, 0.55, 0.55, 0.54, 0.52, 0.50, 0.48], max_steps),
        "south": _repeat([0.64, 0.62, 0.60, 0.60, 0.58, 0.56, 0.54, 0.52], max_steps),
    }
    return TaskScenario(
        task_name="weekday_spike",
        difficulty="easy",
        max_steps=max_steps,
        line_capacity_mw=30,
        neighbor_import_capacity_mw=18,
        objective="Handle a weekday demand spike in the business district without blackouts.",
        zones=zones,
        generators=generators,
        batteries=batteries,
        demand_profile=demand_profile,
        renewable_profile=renewable_profile,
    )


def build_sunset_transition() -> TaskScenario:
    max_steps = 10
    zones = [
        ZoneConfig(zone_id="north", label="North Residential", priority=2, base_demand_mw=56, renewable_capacity_mw=22),
        ZoneConfig(zone_id="east", label="East Tech Park", priority=3, base_demand_mw=64, renewable_capacity_mw=20),
        ZoneConfig(zone_id="south", label="South Industrial", priority=2, base_demand_mw=68, renewable_capacity_mw=16),
        ZoneConfig(zone_id="west", label="West Commercial", priority=2, base_demand_mw=60, renewable_capacity_mw=14),
        ZoneConfig(zone_id="harbor", label="Harbor Logistics", priority=2, base_demand_mw=52, renewable_capacity_mw=18),
        ZoneConfig(zone_id="uptown", label="Uptown Homes", priority=1, base_demand_mw=50, renewable_capacity_mw=28),
        ZoneConfig(zone_id="midtown", label="Midtown Core", priority=3, base_demand_mw=72, renewable_capacity_mw=10),
        ZoneConfig(zone_id="airport", label="Regional Airport", priority=5, critical=True, base_demand_mw=48, renewable_capacity_mw=6),
    ]
    generators = [
        GeneratorConfig(generator_id="gas_n1", zone_id="north", label="North Peaker", min_output_mw=10, max_output_mw=62, ramp_mw_per_step=16, cost_per_mwh=57, initially_enabled=True, initial_output_mw=24),
        GeneratorConfig(generator_id="gas_e1", zone_id="east", label="East Fast Reserve", min_output_mw=8, max_output_mw=58, ramp_mw_per_step=18, cost_per_mwh=60, initially_enabled=False, initial_output_mw=0),
        GeneratorConfig(generator_id="gas_s1", zone_id="south", label="South Turbine", min_output_mw=10, max_output_mw=70, ramp_mw_per_step=18, cost_per_mwh=55, initially_enabled=True, initial_output_mw=30),
        GeneratorConfig(generator_id="diesel_a1", zone_id="airport", label="Airport Backup", min_output_mw=6, max_output_mw=28, ramp_mw_per_step=20, cost_per_mwh=95, startup_cost=12, initially_enabled=False, initial_output_mw=0),
    ]
    batteries = [
        BatteryConfig(battery_id="bat_n1", zone_id="north", label="North Battery", energy_capacity_mwh=70, max_charge_mw=24, max_discharge_mw=24, initial_soc_percent=58),
        BatteryConfig(battery_id="bat_mid1", zone_id="midtown", label="Midtown Battery", energy_capacity_mwh=90, max_charge_mw=30, max_discharge_mw=30, initial_soc_percent=70),
        BatteryConfig(battery_id="bat_air1", zone_id="airport", label="Airport Battery", energy_capacity_mwh=30, max_charge_mw=12, max_discharge_mw=12, initial_soc_percent=80),
    ]
    evening_drop = [0.85, 0.75, 0.62, 0.48, 0.35, 0.22, 0.16, 0.12, 0.10, 0.08]
    wind_variable = [0.55, 0.68, 0.52, 0.62, 0.40, 0.58, 0.44, 0.50, 0.46, 0.42]
    demand_profile = {
        zone.zone_id: _repeat([1.00, 1.04, 1.08, 1.12, 1.16, 1.18, 1.15, 1.10, 1.06, 1.02], max_steps)
        for zone in zones
    }
    renewable_profile = {
        "north": _repeat(evening_drop, max_steps),
        "east": _repeat(evening_drop, max_steps),
        "south": _repeat(wind_variable, max_steps),
        "west": _repeat(evening_drop, max_steps),
        "harbor": _repeat(wind_variable, max_steps),
        "uptown": _repeat(evening_drop, max_steps),
        "midtown": _repeat(evening_drop, max_steps),
        "airport": _repeat([0.45, 0.40, 0.32, 0.26, 0.20, 0.16, 0.12, 0.10, 0.08, 0.08], max_steps),
    }
    return TaskScenario(
        task_name="sunset_transition",
        difficulty="medium",
        max_steps=max_steps,
        line_capacity_mw=42,
        neighbor_import_capacity_mw=32,
        objective="Manage the evening peak as solar drops and wind fluctuates, while preserving service and controlling cost.",
        zones=zones,
        generators=generators,
        batteries=batteries,
        demand_profile=demand_profile,
        renewable_profile=renewable_profile,
    )


def build_heatwave_failure() -> TaskScenario:
    max_steps = 12
    zones = [
        ZoneConfig(zone_id="hospital", label="Metro Hospital", priority=5, critical=True, base_demand_mw=40, renewable_capacity_mw=8),
        ZoneConfig(zone_id="airport", label="International Airport", priority=5, critical=True, base_demand_mw=58, renewable_capacity_mw=10),
        ZoneConfig(zone_id="datacenter", label="Cloud Data Center", priority=5, critical=True, base_demand_mw=52, renewable_capacity_mw=6),
        ZoneConfig(zone_id="north", label="North Residential", priority=2, base_demand_mw=72, renewable_capacity_mw=20),
        ZoneConfig(zone_id="south", label="South Residential", priority=2, base_demand_mw=70, renewable_capacity_mw=18),
        ZoneConfig(zone_id="east", label="East Tech Park", priority=3, base_demand_mw=74, renewable_capacity_mw=14),
        ZoneConfig(zone_id="west", label="West Retail", priority=2, base_demand_mw=68, renewable_capacity_mw=12),
        ZoneConfig(zone_id="industrial", label="Industrial Corridor", priority=3, base_demand_mw=80, renewable_capacity_mw=10),
        ZoneConfig(zone_id="harbor", label="Harbor Logistics", priority=2, base_demand_mw=62, renewable_capacity_mw=16),
        ZoneConfig(zone_id="campus", label="University Campus", priority=1, base_demand_mw=50, renewable_capacity_mw=14),
        ZoneConfig(zone_id="stadium", label="Stadium District", priority=1, base_demand_mw=44, renewable_capacity_mw=8),
        ZoneConfig(zone_id="suburb_a", label="Suburb A", priority=1, base_demand_mw=48, renewable_capacity_mw=18),
        ZoneConfig(zone_id="suburb_b", label="Suburb B", priority=1, base_demand_mw=46, renewable_capacity_mw=18),
        ZoneConfig(zone_id="old_town", label="Old Town", priority=1, base_demand_mw=42, renewable_capacity_mw=10),
        ZoneConfig(zone_id="water", label="Water Treatment", priority=4, critical=True, base_demand_mw=34, renewable_capacity_mw=4),
    ]
    generators = [
        GeneratorConfig(generator_id="main_g1", zone_id="industrial", label="Main Combined Cycle", min_output_mw=30, max_output_mw=130, ramp_mw_per_step=22, cost_per_mwh=48, initially_enabled=True, initial_output_mw=96),
        GeneratorConfig(generator_id="gas_n1", zone_id="north", label="North Peaker", min_output_mw=10, max_output_mw=64, ramp_mw_per_step=18, cost_per_mwh=59, initially_enabled=True, initial_output_mw=34),
        GeneratorConfig(generator_id="gas_s1", zone_id="south", label="South Peaker", min_output_mw=10, max_output_mw=64, ramp_mw_per_step=18, cost_per_mwh=60, initially_enabled=True, initial_output_mw=34),
        GeneratorConfig(generator_id="gas_e1", zone_id="east", label="East Reserve", min_output_mw=8, max_output_mw=58, ramp_mw_per_step=18, cost_per_mwh=62, initially_enabled=False, initial_output_mw=0),
        GeneratorConfig(generator_id="diesel_h1", zone_id="hospital", label="Hospital Backup", min_output_mw=5, max_output_mw=24, ramp_mw_per_step=20, cost_per_mwh=90, startup_cost=15, initially_enabled=False, initial_output_mw=0),
        GeneratorConfig(generator_id="diesel_a1", zone_id="airport", label="Airport Backup", min_output_mw=6, max_output_mw=30, ramp_mw_per_step=20, cost_per_mwh=96, startup_cost=18, initially_enabled=False, initial_output_mw=0),
        GeneratorConfig(generator_id="diesel_d1", zone_id="datacenter", label="Data Center Backup", min_output_mw=6, max_output_mw=32, ramp_mw_per_step=20, cost_per_mwh=94, startup_cost=18, initially_enabled=False, initial_output_mw=0),
    ]
    batteries = [
        BatteryConfig(battery_id="bat_h1", zone_id="hospital", label="Hospital Battery", energy_capacity_mwh=24, max_charge_mw=10, max_discharge_mw=10, initial_soc_percent=88),
        BatteryConfig(battery_id="bat_a1", zone_id="airport", label="Airport Battery", energy_capacity_mwh=36, max_charge_mw=14, max_discharge_mw=14, initial_soc_percent=74),
        BatteryConfig(battery_id="bat_d1", zone_id="datacenter", label="Data Center Battery", energy_capacity_mwh=40, max_charge_mw=14, max_discharge_mw=14, initial_soc_percent=82),
        BatteryConfig(battery_id="bat_ind1", zone_id="industrial", label="Industrial Battery", energy_capacity_mwh=100, max_charge_mw=32, max_discharge_mw=32, initial_soc_percent=68),
        BatteryConfig(battery_id="bat_w1", zone_id="west", label="West Battery", energy_capacity_mwh=70, max_charge_mw=24, max_discharge_mw=24, initial_soc_percent=60),
    ]
    demand_profile = {zone.zone_id: _repeat([1.02, 1.05, 1.10, 1.14, 1.17, 1.20, 1.22, 1.19, 1.16, 1.12, 1.08, 1.04], max_steps) for zone in zones}
    renewable_profile = {zone.zone_id: _repeat([0.72, 0.70, 0.68, 0.64, 0.60, 0.58, 0.55, 0.50, 0.46, 0.42, 0.38, 0.34], max_steps) for zone in zones}
    events = [
        TaskEvent(step=3, event_type="generator_failure", target_id="main_g1", note="Main combined-cycle plant trips offline."),
        TaskEvent(step=4, event_type="fault", target_id="west", duration_steps=2, note="West feeder fault reduces transfer flexibility."),
        TaskEvent(step=6, event_type="line_derate", target_id="global", value=0.75, duration_steps=3, note="Heat derates transmission lines."),
    ]
    return TaskScenario(
        task_name="heatwave_failure",
        difficulty="hard",
        max_steps=max_steps,
        line_capacity_mw=48,
        neighbor_import_capacity_mw=40,
        objective="Protect critical infrastructure during a generator failure and heatwave without cascading blackouts.",
        zones=zones,
        generators=generators,
        batteries=batteries,
        demand_profile=demand_profile,
        renewable_profile=renewable_profile,
        events=events,
    )


TASK_BUILDERS = {
    "weekday_spike": build_weekday_spike,
    "sunset_transition": build_sunset_transition,
    "heatwave_failure": build_heatwave_failure,
}


def get_task(task_name: str) -> TaskScenario:
    if task_name not in TASK_BUILDERS:
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {', '.join(sorted(TASK_BUILDERS))}")
    return TASK_BUILDERS[task_name]()


def list_tasks() -> Dict[str, Dict[str, str]]:
    tasks = {}
    for name, builder in TASK_BUILDERS.items():
        scenario = builder()
        tasks[name] = {
            "difficulty": scenario.difficulty,
            "objective": scenario.objective,
            "weather": WEATHER_BY_TASK[name],
            "time_of_day": TIME_OF_DAY_BY_TASK[name],
        }
    return tasks
