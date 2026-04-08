from __future__ import annotations

import json
from typing import Dict, List

from env.engine import GridLoadBalancerEnv
from env.models import BatteryCommand, GeneratorCommand, GridAction, LoadShedCommand, PowerTransfer
from env.tasks import list_tasks


def heuristic_action(observation) -> GridAction:
    generator_commands: List[GeneratorCommand] = []
    battery_commands: List[BatteryCommand] = []
    transfers: List[PowerTransfer] = []
    shedding: List[LoadShedCommand] = []

    severe_risk = any(zone.critical and zone.unmet_demand_mw > 0.0 for zone in observation.zones)
    low_reserve = observation.reserve_margin_percent < 18.0

    zone_map = {zone.zone_id: zone for zone in observation.zones}
    surplus_by_zone = {
        zone.zone_id: max(0.0, zone.local_generation_mw - zone.demand_mw) for zone in observation.zones
    }

    for generator in observation.generators:
        zone = zone_map[generator.zone_id]
        local_need = max(0.0, zone.demand_mw - zone.local_generation_mw)
        if not generator.available:
            continue
        if zone.critical and (local_need > 0.0 or severe_risk):
            target = min(generator.max_output_mw, max(generator.output_mw, local_need + 8.0))
            generator_commands.append(GeneratorCommand(generator_id=generator.generator_id, enabled=True, target_output_mw=target))
        elif low_reserve and local_need > 6.0:
            target = min(generator.max_output_mw, max(generator.output_mw, local_need + 6.0))
            generator_commands.append(GeneratorCommand(generator_id=generator.generator_id, enabled=True, target_output_mw=target))
        elif not severe_risk and generator.cost_per_mwh > 90 and zone.unmet_demand_mw <= 0.0:
            generator_commands.append(GeneratorCommand(generator_id=generator.generator_id, enabled=False, target_output_mw=0.0))

    for battery in observation.batteries:
        zone = zone_map[battery.zone_id]
        local_deficit = max(0.0, zone.demand_mw - zone.local_generation_mw)
        if zone.critical and battery.available_discharge_mw > 0 and local_deficit > 0:
            battery_commands.append(BatteryCommand(battery_id=battery.battery_id, mode="discharge", power_mw=min(battery.available_discharge_mw, local_deficit)))
        elif low_reserve and battery.available_discharge_mw > 0 and local_deficit > 4.0:
            battery_commands.append(BatteryCommand(battery_id=battery.battery_id, mode="discharge", power_mw=min(battery.available_discharge_mw, local_deficit * 0.8)))
        elif battery.soc_percent < 55 and battery.available_charge_mw > 0 and surplus_by_zone[zone.zone_id] > 0:
            battery_commands.append(BatteryCommand(battery_id=battery.battery_id, mode="charge", power_mw=min(battery.available_charge_mw, surplus_by_zone[zone.zone_id])))

    deficit_targets = sorted(
        observation.zones,
        key=lambda zone: (not zone.critical, -zone.priority, -(zone.demand_mw - zone.local_generation_mw)),
    )
    source_candidates = sorted(observation.zones, key=lambda zone: surplus_by_zone[zone.zone_id], reverse=True)
    for target in deficit_targets:
        remaining = max(0.0, target.demand_mw - target.local_generation_mw)
        if remaining <= 4.0:
            continue
        for source in source_candidates:
            if source.zone_id == target.zone_id:
                continue
            available = surplus_by_zone[source.zone_id]
            if available <= 4.0:
                continue
            amount = min(available, remaining, 20.0)
            transfers.append(PowerTransfer(source_zone_id=source.zone_id, target_zone_id=target.zone_id, mw=amount))
            surplus_by_zone[source.zone_id] -= amount
            remaining -= amount
            if remaining <= 1.0:
                break

    import_caps = {
        "weekday_spike": 18.0,
        "sunset_transition": 30.0,
        "heatwave_failure": 38.0,
        "storm_front_response": 34.0,
        "winter_gas_shortage": 34.0,
    }
    residual_gap = sum(max(0.0, zone.demand_mw - zone.local_generation_mw) for zone in observation.zones)
    neighbor_import_mw = min(import_caps.get(observation.task_name, 20.0), residual_gap) if residual_gap > 0 else 0.0

    if observation.task_name == "heatwave_failure":
        for zone in observation.zones:
            if zone.critical:
                continue
            local_gap = max(0.0, zone.demand_mw - zone.local_generation_mw)
            if local_gap > 18.0:
                shedding.append(LoadShedCommand(zone_id=zone.zone_id, mw=min(local_gap * 0.25, zone.demand_mw * 0.2)))

    return GridAction(
        power_transfers=transfers[:8],
        generator_commands=generator_commands[:8],
        battery_commands=battery_commands[:8],
        load_shed_commands=shedding[:6],
        neighbor_import_mw=neighbor_import_mw,
        reason="heuristic_grid_stabilization",
    )


def run_baseline(task_name: str) -> Dict[str, float]:
    env = GridLoadBalancerEnv(task_name=task_name)
    observation = env.reset(task_name=task_name)
    rewards: List[float] = []
    done = False
    while not done:
        result = env.step(heuristic_action(observation))
        rewards.append(result.reward)
        observation = result.observation
        done = result.done
    state = env.state()
    env.close()
    return {"task_score": round(state.metrics.task_score, 4), "cumulative_reward": round(state.metrics.cumulative_reward, 4), "steps": len(rewards)}


if __name__ == "__main__":
    print(json.dumps({task: run_baseline(task) for task in list_tasks().keys()}, indent=2))
