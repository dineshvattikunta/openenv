from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from .models import (
    GridAction,
    GridMetrics,
    GridObservation,
    GridState,
    GridStepInfo,
    GridStepResult,
    RewardBreakdown,
    ZoneObservation,
    GeneratorObservation,
    BatteryObservation,
)
from .rewards import compute_step_reward, compute_task_score
from .tasks import TIME_OF_DAY_BY_TASK, WEATHER_BY_TASK, get_task
from .utils import clamp


class GridLoadBalancerEnv:
    def __init__(self, task_name: str = "weekday_spike") -> None:
        self.scenario = get_task(task_name)
        self._closed = False
        self._line_capacity_multiplier = 1.0
        self.reserve_margin_percent = 100.0
        self.reset()

    @classmethod
    async def from_docker_image(
        cls,
        image_name: Optional[str] = None,
        task_name: str = "weekday_spike",
    ) -> "GridLoadBalancerEnv":
        return cls(task_name=task_name)

    def reset(self, task_name: Optional[str] = None) -> GridObservation:
        if task_name:
            self.scenario = get_task(task_name)
        self.current_step = 0
        self.metrics = GridMetrics()
        self.weather = WEATHER_BY_TASK[self.scenario.task_name]
        self.time_of_day = TIME_OF_DAY_BY_TASK[self.scenario.task_name]
        self.active_faults: Dict[str, str] = {}
        self.last_info = GridStepInfo(
            valid_action=True,
            reward_breakdown=RewardBreakdown(
                blackout_score=1.0,
                critical_service_score=1.0,
                efficiency_score=1.0,
                cost_score=1.0,
                response_score=1.0,
                penalties=0.0,
                total_reward=0.0,
            ),
        )
        self.zone_configs = {zone.zone_id: deepcopy(zone) for zone in self.scenario.zones}
        self.generator_configs = {gen.generator_id: deepcopy(gen) for gen in self.scenario.generators}
        self.battery_configs = {battery.battery_id: deepcopy(battery) for battery in self.scenario.batteries}
        self.zone_state = {
            zone.zone_id: {
                "demand_mw": zone.base_demand_mw,
                "renewable_output_mw": 0.0,
                "served_mw": 0.0,
                "unmet_demand_mw": 0.0,
                "net_transfer_mw": 0.0,
                "load_shed_mw": 0.0,
                "fault_active": False,
                "status": "normal",
            }
            for zone in self.scenario.zones
        }
        self.generator_state = {
            gen.generator_id: {
                "enabled": gen.initially_enabled,
                "available": True,
                "output_mw": gen.initial_output_mw if gen.initially_enabled else 0.0,
            }
            for gen in self.scenario.generators
        }
        self.battery_state = {
            battery.battery_id: {
                "soc_percent": battery.initial_soc_percent,
                "mode": "idle",
                "power_mw": 0.0,
            }
            for battery in self.scenario.batteries
        }
        self._apply_exogenous_conditions(step_index=1)
        self._recompute_initial_snapshot()
        self.observation = self._build_observation(last_action_error=None)
        return self.observation

    def close(self) -> None:
        self._closed = True

    def state(self) -> GridState:
        return GridState(
            scenario=self.scenario,
            observation=self.observation,
            metrics=self.metrics,
            generator_outputs_mw={gid: data["output_mw"] for gid, data in self.generator_state.items()},
            generator_enabled={gid: data["enabled"] for gid, data in self.generator_state.items()},
            battery_soc_percent={bid: data["soc_percent"] for bid, data in self.battery_state.items()},
            battery_mode={bid: data["mode"] for bid, data in self.battery_state.items()},
            last_info=self.last_info,
        )

    def step(self, action: GridAction | Dict | str | None) -> GridStepResult:
        if self._closed:
            raise RuntimeError("Environment is closed.")

        parsed_action, action_error = self._parse_action(action)
        self.current_step += 1
        self._apply_exogenous_conditions(step_index=self.current_step)
        info = self._apply_action_and_balance(parsed_action, initial_error=action_error)
        done = self.current_step >= self.scenario.max_steps
        self.metrics.task_score = compute_task_score(self.metrics, self.scenario)
        self.last_info = info
        self.observation = self._build_observation(last_action_error=info.last_action_error)
        return GridStepResult(
            observation=self.observation,
            reward=info.reward_breakdown.total_reward,
            done=done,
            info=info,
        )

    def _parse_action(self, action: GridAction | Dict | str | None) -> Tuple[GridAction, Optional[str]]:
        if action is None:
            return GridAction(reason="noop"), None
        if isinstance(action, GridAction):
            return action, None
        try:
            return GridAction.model_validate(action), None
        except ValidationError as exc:
            self.metrics.invalid_action_count += 1
            return GridAction(reason="fallback-noop"), f"invalid_action:{exc.errors()[0]['type']}"

    def _profile_value(self, values: List[float], step_index: int) -> float:
        idx = max(0, min(step_index - 1, len(values) - 1))
        return values[idx]

    def _active_events(self, step_index: int) -> Iterable:
        for event in self.scenario.events:
            if event.step <= step_index < event.step + event.duration_steps:
                yield event

    def _apply_exogenous_conditions(self, step_index: int) -> None:
        self.active_faults = {}
        self._line_capacity_multiplier = 1.0
        for zone_id, zone in self.zone_configs.items():
            self.zone_state[zone_id]["demand_mw"] = zone.base_demand_mw * self._profile_value(
                self.scenario.demand_profile[zone_id], step_index
            )
            self.zone_state[zone_id]["renewable_output_mw"] = zone.renewable_capacity_mw * self._profile_value(
                self.scenario.renewable_profile[zone_id], step_index
            )
            self.zone_state[zone_id]["fault_active"] = False
            self.zone_state[zone_id]["load_shed_mw"] = 0.0
            self.zone_state[zone_id]["net_transfer_mw"] = 0.0

        for event in self._active_events(step_index):
            if event.event_type == "generator_failure" and event.target_id in self.generator_state:
                self.generator_state[event.target_id]["available"] = False
                self.generator_state[event.target_id]["enabled"] = False
                self.generator_state[event.target_id]["output_mw"] = 0.0
                self.active_faults[event.target_id] = event.note or "generator failure"
            elif event.event_type == "fault" and event.target_id in self.zone_state:
                self.zone_state[event.target_id]["fault_active"] = True
                self.active_faults[event.target_id] = event.note or "zone fault"
            elif event.event_type == "line_derate":
                self._line_capacity_multiplier = clamp(event.value or 0.8, 0.4, 1.0)
                self.active_faults["line_derate"] = event.note or "line derating"

    def _apply_action_and_balance(self, action: GridAction, initial_error: Optional[str]) -> GridStepInfo:
        errors: List[str] = []
        valid_action = initial_error is None
        penalty = 0.05 if initial_error else 0.0
        if initial_error:
            errors.append(initial_error)

        transfer_clipped_mw = 0.0
        import_used_mw = 0.0
        total_operating_cost = 0.0
        overloaded_lines = 0

        if action.neighbor_import_mw > self.scenario.neighbor_import_capacity_mw:
            valid_action = False
            errors.append("import_clipped")
            penalty += 0.03
        requested_import_mw = clamp(action.neighbor_import_mw, 0.0, self.scenario.neighbor_import_capacity_mw)

        for cmd in action.load_shed_commands:
            zone = self.zone_configs.get(cmd.zone_id)
            if not zone:
                valid_action = False
                errors.append(f"unknown_zone:{cmd.zone_id}")
                penalty += 0.02
                continue
            max_ratio = 0.1 if zone.critical else 0.35
            demand = self.zone_state[zone.zone_id]["demand_mw"]
            applied = clamp(cmd.mw, 0.0, demand * max_ratio)
            if applied < cmd.mw:
                valid_action = False
                errors.append(f"shed_clipped:{zone.zone_id}")
                penalty += 0.02
            self.zone_state[zone.zone_id]["load_shed_mw"] = min(demand, self.zone_state[zone.zone_id]["load_shed_mw"] + applied)

        toggles = 0
        for gen_id, gen_cfg in self.generator_configs.items():
            state = self.generator_state[gen_id]
            cmd = next((c for c in action.generator_commands if c.generator_id == gen_id), None)
            if not state["available"]:
                state["enabled"] = False
                state["output_mw"] = 0.0
                continue
            prev_enabled = state["enabled"]
            if cmd:
                if cmd.enabled is not None:
                    state["enabled"] = cmd.enabled
                if not state["enabled"]:
                    state["output_mw"] = 0.0
                elif cmd.target_output_mw is not None:
                    target = clamp(cmd.target_output_mw, gen_cfg.min_output_mw, gen_cfg.max_output_mw)
                    state["output_mw"] = target
                    if target != cmd.target_output_mw:
                        valid_action = False
                        errors.append(f"gen_clipped:{gen_id}")
                        penalty += 0.01
            if state["enabled"] and state["output_mw"] <= 0.0:
                state["output_mw"] = gen_cfg.min_output_mw
            if prev_enabled != state["enabled"]:
                toggles += 1
                total_operating_cost += gen_cfg.startup_cost
            total_operating_cost += state["output_mw"] * gen_cfg.cost_per_mwh * (self.scenario.time_step_minutes / 60.0)
        if toggles > 2:
            penalty += 0.02 * (toggles - 2)

        charge_requests: Dict[str, float] = {}
        for battery_id, battery_cfg in self.battery_configs.items():
            state = self.battery_state[battery_id]
            cmd = next((c for c in action.battery_commands if c.battery_id == battery_id), None)
            if not cmd:
                state["mode"] = "idle"
                state["power_mw"] = 0.0
                continue
            if cmd.mode == "charge":
                applied = clamp(cmd.power_mw, 0.0, min(battery_cfg.max_charge_mw, self._max_battery_charge_mw(battery_id)))
                charge_requests[battery_id] = applied
                state["mode"] = "charge"
                state["power_mw"] = applied
                if applied < cmd.power_mw:
                    valid_action = False
                    errors.append(f"charge_clipped:{battery_id}")
                    penalty += 0.01
            elif cmd.mode == "discharge":
                applied = clamp(cmd.power_mw, 0.0, min(battery_cfg.max_discharge_mw, self._max_battery_discharge_mw(battery_id)))
                state["mode"] = "discharge"
                state["power_mw"] = applied
                self._apply_battery_energy_delta(battery_id, applied, discharge=True)
                if applied < cmd.power_mw:
                    valid_action = False
                    errors.append(f"discharge_clipped:{battery_id}")
                    penalty += 0.01
            else:
                state["mode"] = "idle"
                state["power_mw"] = 0.0

        zone_supply_pre: Dict[str, float] = {}
        zone_deficit: Dict[str, float] = {}
        zone_surplus: Dict[str, float] = {}
        demand_after_shed: Dict[str, float] = {}
        for zone_id, zone in self.zone_configs.items():
            renewable = self.zone_state[zone_id]["renewable_output_mw"]
            generator_output = sum(
                self.generator_state[gid]["output_mw"]
                for gid, cfg in self.generator_configs.items()
                if cfg.zone_id == zone_id and self.generator_state[gid]["enabled"] and self.generator_state[gid]["available"]
            )
            battery_discharge = sum(
                self.battery_state[bid]["power_mw"]
                for bid, cfg in self.battery_configs.items()
                if cfg.zone_id == zone_id and self.battery_state[bid]["mode"] == "discharge"
            )
            supply = renewable + generator_output + battery_discharge
            demand = max(0.0, self.zone_state[zone_id]["demand_mw"] - self.zone_state[zone_id]["load_shed_mw"])
            zone_supply_pre[zone_id] = supply
            demand_after_shed[zone_id] = demand
            zone_deficit[zone_id] = max(0.0, demand - supply)
            zone_surplus[zone_id] = max(0.0, supply - demand)

        line_cap = self.scenario.line_capacity_mw * self._line_capacity_multiplier
        for transfer in action.power_transfers:
            if transfer.source_zone_id not in self.zone_configs or transfer.target_zone_id not in self.zone_configs:
                valid_action = False
                errors.append("invalid_transfer_zone")
                penalty += 0.02
                continue
            if transfer.source_zone_id == transfer.target_zone_id:
                valid_action = False
                errors.append("self_transfer")
                penalty += 0.01
                continue
            factor = 0.6 if self.zone_state[transfer.source_zone_id]["fault_active"] or self.zone_state[transfer.target_zone_id]["fault_active"] else 1.0
            actual = clamp(transfer.mw, 0.0, min(line_cap * factor, zone_surplus[transfer.source_zone_id], zone_deficit[transfer.target_zone_id]))
            if actual < transfer.mw:
                transfer_clipped_mw += transfer.mw - actual
                overloaded_lines += 1
                valid_action = False
                errors.append(f"transfer_clipped:{transfer.source_zone_id}->{transfer.target_zone_id}")
                penalty += 0.01
            zone_surplus[transfer.source_zone_id] -= actual
            zone_deficit[transfer.target_zone_id] -= actual
            self.zone_state[transfer.source_zone_id]["net_transfer_mw"] -= actual
            self.zone_state[transfer.target_zone_id]["net_transfer_mw"] += actual

        remaining_import = requested_import_mw
        priority_order = sorted(self.zone_configs.values(), key=lambda zone: (not zone.critical, -zone.priority, zone.zone_id))
        for zone in priority_order:
            deficit = zone_deficit[zone.zone_id]
            if deficit <= 0.0 or remaining_import <= 0.0:
                continue
            used = min(deficit, remaining_import)
            zone_deficit[zone.zone_id] -= used
            remaining_import -= used
            import_used_mw += used
            self.zone_state[zone.zone_id]["net_transfer_mw"] += used
        total_operating_cost += import_used_mw * self.scenario.neighbor_import_cost_per_mwh * (self.scenario.time_step_minutes / 60.0)

        for battery_id, requested in charge_requests.items():
            zone_id = self.battery_configs[battery_id].zone_id
            applied = min(requested, zone_surplus[zone_id])
            self.battery_state[battery_id]["power_mw"] = applied
            if applied <= 0:
                self.battery_state[battery_id]["mode"] = "idle"
                continue
            if applied < requested:
                valid_action = False
                errors.append(f"charge_surplus_limited:{battery_id}")
                penalty += 0.01
            self._apply_battery_energy_delta(battery_id, applied, discharge=False)
            zone_surplus[zone_id] -= applied

        total_demand = 0.0
        total_served = 0.0
        total_critical_demand = 0.0
        total_critical_served = 0.0
        total_available = 0.0
        wasted_energy = 0.0
        critical_zones_at_risk: List[str] = []
        for zone_id, zone in self.zone_configs.items():
            demand = demand_after_shed[zone_id]
            served = max(0.0, demand - zone_deficit[zone_id])
            unmet = max(0.0, zone_deficit[zone_id])
            self.zone_state[zone_id]["served_mw"] = served
            self.zone_state[zone_id]["unmet_demand_mw"] = unmet
            total_demand += demand
            total_served += served
            total_available += zone_supply_pre[zone_id]
            wasted_energy += max(0.0, zone_surplus[zone_id])
            if zone.critical:
                total_critical_demand += demand
                total_critical_served += served
                if unmet > 0.0:
                    critical_zones_at_risk.append(zone_id)
            if unmet <= 0.01:
                status = "normal"
            elif zone.critical or unmet > demand * 0.25:
                status = "critical"
            elif unmet > demand * 0.10:
                status = "overload"
            else:
                status = "warning"
            self.zone_state[zone_id]["status"] = status

        served_ratio = total_served / total_demand if total_demand else 1.0
        critical_served_ratio = total_critical_served / total_critical_demand if total_critical_demand else 1.0
        efficiency_ratio = clamp(1.0 - (wasted_energy / total_available if total_available else 0.0), 0.0, 1.0)
        cost_ratio = clamp(
            total_operating_cost / max(1.0, total_demand * self.scenario.neighbor_import_cost_per_mwh * (self.scenario.time_step_minutes / 60.0)),
            0.0,
            1.0,
        )
        cost_score = 1.0 - cost_ratio
        response_ratio = clamp(
            1.0 - ((len(critical_zones_at_risk) * 0.25) + (overloaded_lines * 0.05) + (transfer_clipped_mw / 100.0)),
            0.0,
            1.0,
        )
        if critical_zones_at_risk:
            penalty += 0.08 * len(critical_zones_at_risk)
        if served_ratio < 0.92:
            penalty += 0.05

        reward = compute_step_reward(served_ratio, critical_served_ratio, efficiency_ratio, cost_score, response_ratio, penalty)
        self._update_metrics(reward, served_ratio, critical_served_ratio, efficiency_ratio, cost_ratio, response_ratio, total_demand, total_served, total_critical_demand, total_critical_served, total_operating_cost, critical_zones_at_risk)
        self.reserve_margin_percent = self._compute_reserve_margin()

        return GridStepInfo(
            valid_action=valid_action,
            last_action_error="|".join(errors) if errors else None,
            transfer_clipped_mw=transfer_clipped_mw,
            import_used_mw=import_used_mw,
            total_operating_cost=total_operating_cost,
            total_wasted_energy_mw=wasted_energy,
            critical_zones_at_risk=critical_zones_at_risk,
            overloaded_lines=overloaded_lines,
            reward_breakdown=RewardBreakdown(
                blackout_score=served_ratio,
                critical_service_score=critical_served_ratio,
                efficiency_score=efficiency_ratio,
                cost_score=cost_score,
                response_score=response_ratio,
                penalties=penalty,
                total_reward=reward,
            ),
        )

    def _update_metrics(
        self,
        reward: float,
        served_ratio: float,
        critical_served_ratio: float,
        efficiency_ratio: float,
        cost_ratio: float,
        response_ratio: float,
        total_demand: float,
        total_served: float,
        total_critical_demand: float,
        total_critical_served: float,
        total_operating_cost: float,
        critical_zones_at_risk: List[str],
    ) -> None:
        self.metrics.reward_history.append(reward)
        self.metrics.cumulative_reward += reward
        steps = len(self.metrics.reward_history)
        self.metrics.average_served_ratio = ((self.metrics.average_served_ratio * (steps - 1)) + served_ratio) / steps
        self.metrics.average_critical_served_ratio = ((self.metrics.average_critical_served_ratio * (steps - 1)) + critical_served_ratio) / steps
        self.metrics.average_efficiency_ratio = ((self.metrics.average_efficiency_ratio * (steps - 1)) + efficiency_ratio) / steps
        self.metrics.average_cost_ratio = ((self.metrics.average_cost_ratio * (steps - 1)) + cost_ratio) / steps
        self.metrics.average_response_ratio = ((self.metrics.average_response_ratio * (steps - 1)) + response_ratio) / steps
        self.metrics.total_unmet_mwh += (total_demand - total_served) * (self.scenario.time_step_minutes / 60.0)
        self.metrics.total_critical_unmet_mwh += (total_critical_demand - total_critical_served) * (self.scenario.time_step_minutes / 60.0)
        self.metrics.total_cost += total_operating_cost
        if total_demand - total_served > 0.0:
            self.metrics.blackouts += 1
        if critical_zones_at_risk:
            self.metrics.critical_blackouts += len(critical_zones_at_risk)

    def _compute_reserve_margin(self) -> float:
        total_demand = sum(self.zone_state[zone_id]["demand_mw"] for zone_id in self.zone_state)
        dispatchable_capacity = self.scenario.neighbor_import_capacity_mw
        dispatch_used = 0.0
        for gen_id, gen_cfg in self.generator_configs.items():
            if self.generator_state[gen_id]["available"]:
                dispatchable_capacity += gen_cfg.max_output_mw
                dispatch_used += self.generator_state[gen_id]["output_mw"]
        for battery_id, battery_cfg in self.battery_configs.items():
            dispatchable_capacity += min(battery_cfg.max_discharge_mw, self._max_battery_discharge_mw(battery_id))
            if self.battery_state[battery_id]["mode"] == "discharge":
                dispatch_used += self.battery_state[battery_id]["power_mw"]
        reserve = max(0.0, dispatchable_capacity - dispatch_used)
        return clamp((reserve / total_demand) * 100.0 if total_demand else 100.0, 0.0, 100.0)

    def _max_battery_charge_mw(self, battery_id: str) -> float:
        cfg = self.battery_configs[battery_id]
        remaining_mwh = cfg.energy_capacity_mwh * (1.0 - (self.battery_state[battery_id]["soc_percent"] / 100.0))
        return remaining_mwh / (self.scenario.time_step_minutes / 60.0)

    def _max_battery_discharge_mw(self, battery_id: str) -> float:
        cfg = self.battery_configs[battery_id]
        available_mwh = cfg.energy_capacity_mwh * (self.battery_state[battery_id]["soc_percent"] / 100.0)
        return available_mwh / (self.scenario.time_step_minutes / 60.0)

    def _apply_battery_energy_delta(self, battery_id: str, power_mw: float, discharge: bool) -> None:
        cfg = self.battery_configs[battery_id]
        delta_mwh = power_mw * (self.scenario.time_step_minutes / 60.0)
        current_mwh = cfg.energy_capacity_mwh * (self.battery_state[battery_id]["soc_percent"] / 100.0)
        if discharge:
            current_mwh = max(0.0, current_mwh - delta_mwh)
        else:
            current_mwh = min(cfg.energy_capacity_mwh, current_mwh + (delta_mwh * cfg.round_trip_efficiency))
        self.battery_state[battery_id]["soc_percent"] = (current_mwh / cfg.energy_capacity_mwh) * 100.0

    def _build_observation(self, last_action_error: Optional[str]) -> GridObservation:
        zones = []
        for zone_id, zone in self.zone_configs.items():
            battery_socs = [self.battery_state[bid]["soc_percent"] for bid, cfg in self.battery_configs.items() if cfg.zone_id == zone_id]
            local_generation = self.zone_state[zone_id]["renewable_output_mw"] + sum(
                self.generator_state[gid]["output_mw"]
                for gid, cfg in self.generator_configs.items()
                if cfg.zone_id == zone_id and self.generator_state[gid]["enabled"] and self.generator_state[gid]["available"]
            )
            zones.append(
                ZoneObservation(
                    zone_id=zone_id,
                    label=zone.label,
                    priority=zone.priority,
                    critical=zone.critical,
                    demand_mw=round(self.zone_state[zone_id]["demand_mw"], 2),
                    served_mw=round(self.zone_state[zone_id]["served_mw"], 2),
                    unmet_demand_mw=round(self.zone_state[zone_id]["unmet_demand_mw"], 2),
                    local_generation_mw=round(local_generation, 2),
                    renewable_output_mw=round(self.zone_state[zone_id]["renewable_output_mw"], 2),
                    battery_soc_percent=round(sum(battery_socs) / len(battery_socs), 2) if battery_socs else 0.0,
                    net_transfer_mw=round(self.zone_state[zone_id]["net_transfer_mw"], 2),
                    load_shed_mw=round(self.zone_state[zone_id]["load_shed_mw"], 2),
                    fault_active=self.zone_state[zone_id]["fault_active"],
                    status=self.zone_state[zone_id]["status"],
                )
            )

        generators = [
            GeneratorObservation(
                generator_id=gen_id,
                zone_id=cfg.zone_id,
                label=cfg.label,
                enabled=self.generator_state[gen_id]["enabled"],
                available=self.generator_state[gen_id]["available"],
                output_mw=round(self.generator_state[gen_id]["output_mw"], 2),
                max_output_mw=cfg.max_output_mw,
                cost_per_mwh=cfg.cost_per_mwh,
            )
            for gen_id, cfg in self.generator_configs.items()
        ]
        batteries = [
            BatteryObservation(
                battery_id=bid,
                zone_id=cfg.zone_id,
                label=cfg.label,
                soc_percent=round(self.battery_state[bid]["soc_percent"], 2),
                mode=self.battery_state[bid]["mode"],
                power_mw=round(self.battery_state[bid]["power_mw"], 2),
                available_charge_mw=round(min(cfg.max_charge_mw, self._max_battery_charge_mw(bid)), 2),
                available_discharge_mw=round(min(cfg.max_discharge_mw, self._max_battery_discharge_mw(bid)), 2),
            )
            for bid, cfg in self.battery_configs.items()
        ]
        critical_risk = [zone.zone_id for zone in zones if zone.critical and zone.unmet_demand_mw > 0.0]
        summary = f"Step {self.current_step}/{self.scenario.max_steps}. Reserve margin {self.reserve_margin_percent:.1f}%. Critical risk: {', '.join(critical_risk) if critical_risk else 'none'}."
        return GridObservation(
            task_name=self.scenario.task_name,
            benchmark=self.scenario.benchmark,
            step=self.current_step,
            max_steps=self.scenario.max_steps,
            time_of_day=self.time_of_day,
            weather=self.weather,
            reserve_margin_percent=round(self.reserve_margin_percent, 2),
            active_faults=list(self.active_faults.keys()),
            overloaded_lines=self.last_info.overloaded_lines if self.current_step else 0,
            critical_zones_at_risk=critical_risk,
            last_action_error=last_action_error,
            zones=zones,
            generators=generators,
            batteries=batteries,
            summary=summary,
        )

    def _recompute_initial_snapshot(self) -> None:
        total_demand = 0.0
        total_dispatchable = self.scenario.neighbor_import_capacity_mw
        used_dispatchable = 0.0

        for zone_id in self.zone_configs:
            renewable = self.zone_state[zone_id]["renewable_output_mw"]
            generator_output = sum(
                self.generator_state[gid]["output_mw"]
                for gid, cfg in self.generator_configs.items()
                if cfg.zone_id == zone_id and self.generator_state[gid]["enabled"] and self.generator_state[gid]["available"]
            )
            battery_discharge = 0.0
            local_supply = renewable + generator_output + battery_discharge
            demand = self.zone_state[zone_id]["demand_mw"]
            served = min(demand, local_supply)
            unmet = max(0.0, demand - local_supply)
            self.zone_state[zone_id]["served_mw"] = served
            self.zone_state[zone_id]["unmet_demand_mw"] = unmet
            self.zone_state[zone_id]["net_transfer_mw"] = 0.0
            if unmet <= 0.01:
                status = "normal"
            elif unmet > demand * 0.25:
                status = "critical" if self.zone_configs[zone_id].critical else "overload"
            else:
                status = "warning"
            self.zone_state[zone_id]["status"] = status
            total_demand += demand

        for gen_id, gen_cfg in self.generator_configs.items():
            if self.generator_state[gen_id]["available"]:
                total_dispatchable += gen_cfg.max_output_mw
                used_dispatchable += self.generator_state[gen_id]["output_mw"]

        for battery_id, battery_cfg in self.battery_configs.items():
            total_dispatchable += min(battery_cfg.max_discharge_mw, self._max_battery_discharge_mw(battery_id))

        reserve = max(0.0, total_dispatchable - used_dispatchable)
        self.reserve_margin_percent = clamp((reserve / total_demand) * 100.0 if total_demand else 100.0, 0.0, 100.0)
