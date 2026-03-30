"""OpenAI planner backend configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import PlanningConfig


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    api_key: str | None
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    timeout_seconds: float = 60.0
    retry_count: int = 2
    max_output_tokens: int = 4000

    @classmethod
    def from_env(cls) -> "OpenAIPlannerConfig":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
            reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "medium"),
            timeout_seconds=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")),
            retry_count=int(os.environ.get("OPENAI_RETRY_COUNT", "2")),
            max_output_tokens=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "4000")),
        )

    @classmethod
    def from_planning_config(cls, planning_config: PlanningConfig) -> "OpenAIPlannerConfig":
        env_config = cls.from_env()
        return cls(
            api_key=env_config.api_key,
            model=planning_config.openai_model or env_config.model,
            reasoning_effort=planning_config.openai_reasoning_effort or env_config.reasoning_effort,
            timeout_seconds=planning_config.openai_timeout_seconds or env_config.timeout_seconds,
            retry_count=planning_config.openai_retry_count if planning_config.openai_retry_count is not None else env_config.retry_count,
            max_output_tokens=planning_config.openai_max_output_tokens if planning_config.openai_max_output_tokens is not None else env_config.max_output_tokens,
        )
