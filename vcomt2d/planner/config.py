"""Planner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PlanningConfig:
    max_retries: int = 2
    strict_validation: bool = True
    allow_repair: bool = True
    enable_semantic_checks: bool = True
    deterministic_seed: int = 0
    default_timeout_ticks: Dict[str, int] = field(
        default_factory=lambda: {
            "approach": 8,
            "hold": 5,
            "push": 12,
            "search": 10,
            "converge": 8,
            "handoff": 8,
        }
    )
    animation_enabled: bool = True
    animation_output_dir: str = "outputs/animations"
    animation_fps: int = 4
    mp4_writer_backend: str = "ffmpeg"
    save_backend_artifacts: bool = True
    openai_model: Optional[str] = None
    openai_reasoning_effort: Optional[str] = None
    openai_timeout_seconds: Optional[float] = None
    openai_retry_count: Optional[int] = None
    openai_max_output_tokens: Optional[int] = None
