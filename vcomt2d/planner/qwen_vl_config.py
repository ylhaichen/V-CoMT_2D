"""Configuration for the local Qwen2.5-VL planner backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .config import PlanningConfig


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QwenVLPlannerConfig:
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 768
    temperature: float = 0.0
    device_map: str = "auto"
    local_files_only: bool = False
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    use_scene_image: bool = False
    attn_implementation: Optional[str] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None

    @classmethod
    def from_env(cls) -> "QwenVLPlannerConfig":
        return cls(
            model=os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
            max_new_tokens=int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "768")),
            temperature=float(os.environ.get("QWEN_VL_TEMPERATURE", "0.0")),
            device_map=os.environ.get("QWEN_VL_DEVICE_MAP", "auto"),
            local_files_only=_env_flag("QWEN_VL_LOCAL_FILES_ONLY", False),
            load_in_4bit=_env_flag("QWEN_VL_LOAD_IN_4BIT", True),
            load_in_8bit=_env_flag("QWEN_VL_LOAD_IN_8BIT", False),
            use_scene_image=_env_flag("QWEN_VL_USE_SCENE_IMAGE", False),
            attn_implementation=os.environ.get("QWEN_VL_ATTN_IMPLEMENTATION"),
            min_pixels=int(os.environ["QWEN_VL_MIN_PIXELS"]) if os.environ.get("QWEN_VL_MIN_PIXELS") else None,
            max_pixels=int(os.environ["QWEN_VL_MAX_PIXELS"]) if os.environ.get("QWEN_VL_MAX_PIXELS") else None,
        )

    @classmethod
    def from_planning_config(cls, planning_config: PlanningConfig) -> "QwenVLPlannerConfig":
        env_config = cls.from_env()
        return cls(
            model=planning_config.qwen_vl_model or env_config.model,
            max_new_tokens=planning_config.qwen_vl_max_new_tokens or env_config.max_new_tokens,
            temperature=planning_config.qwen_vl_temperature if planning_config.qwen_vl_temperature is not None else env_config.temperature,
            device_map=planning_config.qwen_vl_device_map or env_config.device_map,
            local_files_only=planning_config.qwen_vl_local_files_only if planning_config.qwen_vl_local_files_only is not None else env_config.local_files_only,
            load_in_4bit=planning_config.qwen_vl_load_in_4bit if planning_config.qwen_vl_load_in_4bit is not None else env_config.load_in_4bit,
            load_in_8bit=planning_config.qwen_vl_load_in_8bit if planning_config.qwen_vl_load_in_8bit is not None else env_config.load_in_8bit,
            use_scene_image=planning_config.qwen_vl_use_scene_image if planning_config.qwen_vl_use_scene_image is not None else env_config.use_scene_image,
            attn_implementation=planning_config.qwen_vl_attn_implementation or env_config.attn_implementation,
            min_pixels=planning_config.qwen_vl_min_pixels if planning_config.qwen_vl_min_pixels is not None else env_config.min_pixels,
            max_pixels=planning_config.qwen_vl_max_pixels if planning_config.qwen_vl_max_pixels is not None else env_config.max_pixels,
        )
