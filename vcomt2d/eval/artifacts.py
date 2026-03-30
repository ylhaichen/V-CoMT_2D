"""Artifact serialization helpers for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from vcomt2d.core.logging import ExecutionTrace
from vcomt2d.fsm.schema import FSMPlan
from .models import EvalArtifactPaths, EvalRunSummary


def prepare_run_dir(base_output_dir: str, task_key: str, run_id: str) -> Path:
    run_dir = Path(base_output_dir) / task_key / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(run_dir: Path, filename: str, payload) -> str:
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(path)


def write_plan(run_dir: Path, filename: str, plan: Optional[FSMPlan]) -> Optional[str]:
    if plan is None:
        return None
    return write_json(run_dir, filename, plan.to_dict())


def write_trace(run_dir: Path, trace: Optional[ExecutionTrace]) -> Optional[str]:
    if trace is None:
        return None
    return write_json(run_dir, "trace.json", trace.to_dict())


def write_summary(run_dir: Path, summary: EvalRunSummary) -> str:
    return write_json(run_dir, "summary.json", summary.to_dict())


def write_logs(run_dir: Path, lines: Iterable[str]) -> str:
    path = run_dir / "logs.txt"
    payload = "\n".join(lines).rstrip() + "\n"
    path.write_text(payload, encoding="utf-8")
    return str(path)
