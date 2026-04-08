from __future__ import annotations

import importlib
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
OPENENV_PATH = ROOT / "openenv.yaml"
INFERENCE_PATH = ROOT / "inference.py"
EXPECTED_TASK_COUNT = 5
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def _load_openenv() -> dict:
    return yaml.safe_load(OPENENV_PATH.read_text(encoding="utf-8"))


def _import_grader(path: str):
    module_name, fn_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def test_task_count_and_graders() -> None:
    cfg = _load_openenv()
    tasks = cfg.get("tasks", [])
    assert len(tasks) == EXPECTED_TASK_COUNT, (
        f"Expected {EXPECTED_TASK_COUNT} tasks, found {len(tasks)}"
    )
    for task in tasks:
        assert task.get("grader"), f"Task {task.get('id')} missing grader"


def test_grader_range_and_edges() -> None:
    cfg = _load_openenv()
    for task in cfg["tasks"]:
        grader = _import_grader(task["grader"])
        default_score = float(grader())
        low_score = float(grader(score=0.0))
        high_score = float(grader(score=1.0))

        assert MIN_SCORE <= default_score <= MAX_SCORE, (
            f"{task['id']} default score out of range: {default_score}"
        )
        assert low_score == MIN_SCORE, (
            f"{task['id']} low clamp expected {MIN_SCORE}, got {low_score}"
        )
        assert high_score == MAX_SCORE, (
            f"{task['id']} high clamp expected {MAX_SCORE}, got {high_score}"
        )


def test_inference_api_base_url_check() -> None:
    text = INFERENCE_PATH.read_text(encoding="utf-8")
    has_env_get = bool(
        re.search(
            r"API_BASE_URL\s*=\s*os\.getenv\(\s*['\"]API_BASE_URL['\"]",
            text,
        )
    )
    has_default = "https://router.huggingface.co/v1" in text
    assert has_env_get, "inference.py missing API_BASE_URL os.getenv lookup"
    assert has_default, "inference.py missing API_BASE_URL default value"

