from __future__ import annotations

import json
from typing import Any, Dict

from .models import GridAction


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def action_to_log_string(action: GridAction) -> str:
    parts = []
    if action.power_transfers:
        parts.append(
            "transfers="
            + ",".join(f"{t.source_zone_id}->{t.target_zone_id}:{t.mw:.1f}" for t in action.power_transfers[:4])
        )
    if action.generator_commands:
        parts.append(
            "gens="
            + ",".join(
                f"{c.generator_id}:{'on' if c.enabled is not False else 'off'}:{(c.target_output_mw or 0.0):.1f}"
                for c in action.generator_commands[:4]
            )
        )
    if action.battery_commands:
        parts.append("batt=" + ",".join(f"{c.battery_id}:{c.mode}:{c.power_mw:.1f}" for c in action.battery_commands[:4]))
    if action.load_shed_commands:
        parts.append("shed=" + ",".join(f"{c.zone_id}:{c.mw:.1f}" for c in action.load_shed_commands[:4]))
    if action.neighbor_import_mw:
        parts.append(f"import={action.neighbor_import_mw:.1f}")
    if action.reason:
        reason = action.reason.replace("\n", " ").replace("\r", " ").strip()
        parts.append(f"reason={reason[:80]}")
    return ";".join(parts) if parts else "noop"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])
