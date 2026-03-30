"""Local Qwen2.5-VL planner backend for structured multimodal planning."""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from vcomt2d.core.types import to_serializable
from vcomt2d.core.skills import VALID_SKILLS
from vcomt2d.fsm.schema import FSMPlan
from vcomt2d.viz.renderer import save_world_state_snapshot

from .backends import PlannerBackend, PlannerBackendError
from .models import BackendPlanCandidate, PlanningContext
from .qwen_vl_config import QwenVLPlannerConfig
from .structured_output import SUPPORTED_CONDITIONS, build_planner_system_prompt, parse_candidate_plan_text


class QwenVLLocalEngine:
    """Lazy local inference engine for Qwen2.5-VL via Transformers."""

    engine_name = "transformers_local"

    def __init__(self):
        self._model = None
        self._processor = None
        self._runtime_config: Optional[QwenVLPlannerConfig] = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_payload: Dict[str, Any],
        user_prompt_text: str,
        runtime_config: QwenVLPlannerConfig,
        scene_image_path: str | None = None,
    ) -> Dict[str, Any]:
        torch = self._import_torch()
        model, processor = self._load_runtime(runtime_config)
        messages = self._build_messages(system_prompt, user_prompt_text, scene_image_path)
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._prepare_inputs(processor, chat_text, messages, scene_image_path)
        input_device = self._resolve_input_device(model)
        inputs = inputs.to(input_device)

        generation_kwargs = {
            "max_new_tokens": runtime_config.max_new_tokens,
            "do_sample": runtime_config.temperature > 0.0,
        }
        if runtime_config.temperature > 0.0:
            generation_kwargs["temperature"] = runtime_config.temperature

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.input_ids
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "messages": to_serializable(messages),
            "chat_text": chat_text,
            "generation_kwargs": generation_kwargs,
            "output_text": output_text,
            "raw_response": {
                "generated_text": output_text,
                "chat_text": chat_text,
                "messages": to_serializable(messages),
                "generation_kwargs": generation_kwargs,
                "scene_image_path": scene_image_path,
            },
        }

    def _load_runtime(self, runtime_config: QwenVLPlannerConfig):
        if self._model is not None and self._processor is not None and self._runtime_config == runtime_config:
            return self._model, self._processor

        torch = self._import_torch()
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise PlannerBackendError(
                "qwen_vl_dependencies_missing",
                debug_info={
                    "required_packages": ["torch", "transformers", "accelerate"],
                    "optional_packages": ["bitsandbytes", "qwen-vl-utils"],
                    "import_error": {
                        "exception_type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            ) from exc

        quantization_config = None
        if runtime_config.load_in_4bit or runtime_config.load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=runtime_config.load_in_4bit,
                    load_in_8bit=runtime_config.load_in_8bit,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=getattr(torch, "float16"),
                )
            except Exception as exc:  # noqa: BLE001 - backend must surface missing quantization support cleanly
                raise PlannerBackendError(
                    "qwen_vl_quantization_unavailable",
                    debug_info={
                        "load_in_4bit": runtime_config.load_in_4bit,
                        "load_in_8bit": runtime_config.load_in_8bit,
                        "quantization_error": {
                            "exception_type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    },
                ) from exc

        load_kwargs: Dict[str, Any] = {
            "device_map": runtime_config.device_map,
            "torch_dtype": "auto",
            "local_files_only": runtime_config.local_files_only,
        }
        if runtime_config.attn_implementation:
            load_kwargs["attn_implementation"] = runtime_config.attn_implementation
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        model_source = self._resolve_model_source(runtime_config)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_source, **load_kwargs)
            processor_kwargs = self._build_processor_kwargs(runtime_config)
            processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
        except Exception as exc:  # noqa: BLE001 - convert to structured backend error
            fallback_result = self._try_local_cache_fallback(
                runtime_config,
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                load_kwargs,
                model_source,
                exc,
            )
            if fallback_result is None:
                raise PlannerBackendError(
                    "qwen_vl_model_load_failed",
                    debug_info={
                        "model_name": runtime_config.model,
                        "model_source": str(model_source),
                        "device_map": runtime_config.device_map,
                        "local_files_only": runtime_config.local_files_only,
                        "model_load_error": {
                            "exception_type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    },
                ) from exc
            model, processor = fallback_result

        self._model = model
        self._processor = processor
        self._runtime_config = runtime_config
        return model, processor

    def _build_processor_kwargs(self, runtime_config: QwenVLPlannerConfig) -> Dict[str, Any]:
        processor_kwargs: Dict[str, Any] = {
            "local_files_only": runtime_config.local_files_only,
        }
        if runtime_config.min_pixels is not None:
            processor_kwargs["min_pixels"] = runtime_config.min_pixels
        if runtime_config.max_pixels is not None:
            processor_kwargs["max_pixels"] = runtime_config.max_pixels
        return processor_kwargs

    def _try_local_cache_fallback(self, runtime_config, model_cls, processor_cls, load_kwargs, model_source, original_exc):
        if runtime_config.local_files_only or not self._looks_like_network_error(original_exc):
            return None
        fallback_source = self._resolve_local_snapshot_path(runtime_config.model)
        if fallback_source is None:
            return None
        fallback_load_kwargs = {**load_kwargs, "local_files_only": True}
        fallback_processor_kwargs = {**self._build_processor_kwargs(runtime_config), "local_files_only": True}
        try:
            model = model_cls.from_pretrained(fallback_source, **fallback_load_kwargs)
            processor = processor_cls.from_pretrained(fallback_source, **fallback_processor_kwargs)
            return model, processor
        except Exception:
            return None

    def _resolve_model_source(self, runtime_config: QwenVLPlannerConfig) -> str:
        if runtime_config.local_files_only:
            local_snapshot = self._resolve_local_snapshot_path(runtime_config.model)
            if local_snapshot is not None:
                return local_snapshot
        return runtime_config.model

    def _resolve_local_snapshot_path(self, model_name: str) -> str | None:
        if model_name.startswith("/") or model_name.startswith("./") or model_name.startswith("../"):
            return model_name
        repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        refs_main = repo_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            candidate = snapshots_dir / revision
            if candidate.exists():
                return str(candidate)
        snapshot_paths = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if not snapshot_paths:
            return None
        return str(snapshot_paths[-1])

    def _prepare_inputs(self, processor, chat_text: str, messages: list[Dict[str, Any]], scene_image_path: str | None):
        if scene_image_path is None:
            return processor(text=[chat_text], padding=True, return_tensors="pt")

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise PlannerBackendError(
                "qwen_vl_utils_missing_for_scene_image",
                debug_info={
                    "scene_image_path": scene_image_path,
                    "required_package": "qwen-vl-utils",
                    "vision_import_error": {
                        "exception_type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            ) from exc

        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _build_messages(self, system_prompt: str, user_prompt_text: str, scene_image_path: str | None) -> list[Dict[str, Any]]:
        user_content: list[Dict[str, Any]] = []
        if scene_image_path is not None:
            user_content.append({"type": "image", "image": f"file://{scene_image_path}"})
        user_content.append({"type": "text", "text": user_prompt_text})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _resolve_input_device(self, model):
        device = getattr(model, "device", None)
        if device is not None:
            return device
        try:
            return next(model.parameters()).device
        except Exception:  # noqa: BLE001 - defensive fallback for accelerate device maps
            return "cpu"

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise PlannerBackendError("qwen_vl_torch_missing", debug_info={"required_package": "torch"}) from exc
        return torch

    def _looks_like_network_error(self, exc: Exception) -> bool:
        text = f"{exc.__class__.__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "temporary failure in name resolution",
                "failed to resolve",
                "connection error",
                "network is unreachable",
                "max retries exceeded",
                "offline",
                "timed out",
            )
        )


class QwenVLPlannerBackend(PlannerBackend):
    """Local Qwen2.5-VL candidate generator for 8 GB VRAM oriented experiments."""

    backend_name = "qwen_vl"

    def __init__(
        self,
        config: Optional[QwenVLPlannerConfig] = None,
        engine: Optional[QwenVLLocalEngine] = None,
    ):
        self.config = config
        self.engine = engine or QwenVLLocalEngine()

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        runtime_config = self.config or QwenVLPlannerConfig.from_planning_config(context.request.planning_config)
        system_prompt = build_planner_system_prompt()
        user_payload = self._build_prompt_payload(context)
        user_prompt_text = self._build_user_prompt_text(user_payload)
        scene_image_path = self._maybe_render_scene_image(context, runtime_config)
        request_payload = {
            "engine_name": getattr(self.engine, "engine_name", "local"),
            "model_name": runtime_config.model,
            "use_scene_image": runtime_config.use_scene_image,
            "scene_image_path": scene_image_path,
            "generation_config": {
                "max_new_tokens": runtime_config.max_new_tokens,
                "temperature": runtime_config.temperature,
                "device_map": runtime_config.device_map,
                "load_in_4bit": runtime_config.load_in_4bit,
                "load_in_8bit": runtime_config.load_in_8bit,
                "attn_implementation": runtime_config.attn_implementation,
                "min_pixels": runtime_config.min_pixels,
                "max_pixels": runtime_config.max_pixels,
            },
        }
        backend_debug = {
            "api": "local_transformers",
            "engine_name": getattr(self.engine, "engine_name", "local"),
            "model_name": runtime_config.model,
            "system_prompt": system_prompt,
            "prompt_payload": user_payload,
            "prompt_text": user_prompt_text,
            "request_payload": request_payload,
            "scene_image_path": scene_image_path,
        }

        try:
            generation = self.engine.generate(
                system_prompt=system_prompt,
                user_payload=user_payload,
                user_prompt_text=user_prompt_text,
                runtime_config=runtime_config,
                scene_image_path=scene_image_path,
            )
        except PlannerBackendError as exc:
            error_debug = exc.debug_info if isinstance(exc.debug_info, dict) else {}
            backend_debug.update(error_debug)
            raise PlannerBackendError(str(exc), debug_info=backend_debug) from exc
        except Exception as exc:  # noqa: BLE001 - unknown local runtime error
            backend_debug["runtime_error"] = {
                "exception_type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            raise PlannerBackendError("qwen_vl_inference_failed", debug_info=backend_debug) from exc

        raw_response = generation.get("raw_response")
        output_text = generation.get("output_text", "")
        if not output_text:
            backend_debug["raw_response"] = raw_response
            raise PlannerBackendError("qwen_vl_response_missing_text", debug_info=backend_debug)

        try:
            candidate_plan = parse_candidate_plan_text(output_text)
        except json.JSONDecodeError as exc:
            backend_debug["raw_response"] = raw_response
            backend_debug["response_text"] = output_text
            if self._looks_like_prompt_echo(output_text):
                raise PlannerBackendError("qwen_vl_prompt_echo_or_truncated_output", debug_info=backend_debug) from exc
            raise PlannerBackendError(f"qwen_vl_response_not_json:{exc}", debug_info=backend_debug) from exc

        try:
            plan = FSMPlan.from_dict(candidate_plan)
        except Exception as exc:  # noqa: BLE001 - surface parser failure through planner backend error
            backend_debug["raw_response"] = raw_response
            backend_debug["candidate_plan"] = candidate_plan
            raise PlannerBackendError(f"qwen_vl_candidate_parse_failed:{exc}", debug_info=backend_debug) from exc

        return BackendPlanCandidate(
            backend_name=self.backend_name,
            model_name=runtime_config.model,
            plan=plan,
            debug_info={
                **backend_debug,
                "raw_response": raw_response,
                "candidate_plan": candidate_plan,
                "messages": generation.get("messages"),
                "chat_text": generation.get("chat_text"),
            },
        )

    def _maybe_render_scene_image(self, context: PlanningContext, runtime_config: QwenVLPlannerConfig) -> str | None:
        if not runtime_config.use_scene_image:
            return None
        temp_dir = Path(tempfile.mkdtemp(prefix="vcomt2d_qwen_scene_", dir="/tmp"))
        output_path = temp_dir / f"{context.request.request_id}_scene.png"
        return save_world_state_snapshot(context.request.world_state, str(output_path), title=f"{context.request.request_id} planner input")

    def _build_prompt_payload(self, context: PlanningContext) -> Dict[str, Any]:
        return {
            "request_id": context.request.request_id,
            "instruction": context.request.user_instruction,
            "task_type": None if context.intent.task_type is None else context.intent.task_type.value,
            "scene_facts": context.scene_facts.to_dict(),
            "role_assignment": context.roles.to_dict(),
            "world_state": context.request.world_state.to_dict(),
        }

    def _build_user_prompt_text(self, payload: Dict[str, Any]) -> str:
        compact_scene = json.dumps(payload["scene_facts"], separators=(",", ":"), ensure_ascii=True)
        compact_roles = json.dumps(payload["role_assignment"], separators=(",", ":"), ensure_ascii=True)
        compact_world = json.dumps(payload["world_state"], separators=(",", ":"), ensure_ascii=True)
        return (
            "Transform the planning input into a NEW FSM plan.\n"
            "Do not copy or restate the planning input fields.\n"
            "Do not output request_id, instruction, scene_facts, role_assignment, or world_state.\n"
            "Return only one JSON object, with no markdown fences.\n\n"
            f"Task instruction: {payload['instruction']}\n"
            f"Task type hint: {payload['task_type']}\n"
            f"Scene facts: {compact_scene}\n"
            f"Role assignment hint: {compact_roles}\n"
            f"World state: {compact_world}\n\n"
            "Allowed skills:\n"
            f"{sorted(VALID_SKILLS)}\n"
            "Allowed condition kinds:\n"
            f"{SUPPORTED_CONDITIONS}\n\n"
            "Output requirements:\n"
            "- top-level keys must be exactly: task_description, reasoning_summary, fsm\n"
            "- every non-terminal state must have robot_a and robot_b actions\n"
            "- include a success terminal state with status='success'\n"
            "- include a failure terminal state with status='failure'\n"
            "- every non-terminal state must include timeout coverage to S_FAIL\n"
            "- use readable state names such as S0_APPROACH, S1_HOLD, S_DONE, S_FAIL\n"
            "- both robots must have meaningful collaborative roles\n\n"
            "Return JSON using this template:\n"
            "{\n"
            '  "task_description": "short task description",\n'
            '  "reasoning_summary": "short factual planning summary",\n'
            '  "fsm": {\n'
            '    "initial_state": "S0_APPROACH",\n'
            '    "states": [\n'
            "      {\n"
            '        "state_id": "S0_APPROACH",\n'
            '        "robot_a": {"skill": "MOVE_TO", "params": {"target_id": "some_target"}},\n'
            '        "robot_b": {"skill": "HOLD_POSITION", "params": {"ticks": 1}},\n'
            '        "transitions": [\n'
            '          {"to": "S1_NEXT", "condition": {"kind": "all_actions_done", "args": {}}, "notes": "advance"},\n'
            '          {"to": "S_FAIL", "condition": {"kind": "timeout", "args": {"ticks": 8}}, "notes": "timeout"}\n'
            "        ],\n"
            '        "terminal": false,\n'
            '        "status": null,\n'
            '        "notes": "state purpose"\n'
            "      },\n"
            '      {"state_id": "S_DONE", "robot_a": null, "robot_b": null, "transitions": [], "terminal": true, "status": "success", "notes": "done"},\n'
            '      {"state_id": "S_FAIL", "robot_a": null, "robot_b": null, "transitions": [], "terminal": true, "status": "failure", "notes": "failure"}\n'
            "    ]\n"
            "  }\n"
            "}\n"
        )

    def _looks_like_prompt_echo(self, output_text: str) -> bool:
        lowered = output_text.lower()
        return (
            '"request_id"' in lowered
            or '"instruction"' in lowered
            or '"world_state"' in lowered
            or '"scene_facts"' in lowered
            or '"task_type_hint"' in lowered
        ) and '"task_description"' not in lowered
