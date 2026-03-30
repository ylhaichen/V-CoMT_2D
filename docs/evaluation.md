# Evaluation

## Purpose

The evaluation harness turns the planner/executor stack into a benchmark-ready subsystem.

It supports:

- deterministic batch runs
- GPT-backed batch runs
- local `Qwen2.5-VL-3B-Instruct` batch runs
- per-run artifacts
- machine-readable summaries
- animation export during evaluation
- failure analysis through stored traces and logs

For model-based backends, the harness also preserves enough artifacts to distinguish:

- raw model generation success or failure
- structural / semantic repair activity
- deterministic resynthesis fallback activity

## Main Entry Point

```bash
python3 scripts/run_eval.py --tasks door herding search relay --runs 1
```

## Request Controls

`run_eval.py` supports:

- `--tasks`
  - select one or more task families
- `--runs`
  - deterministic repetitions per task
- `--backend`
  - choose `deterministic`, `gpt`, `qwen_vl`, `llm_stub`, or `vlm_stub`
- `--model`
  - override the model name for GPT or `Qwen`
- `--reasoning-effort`
  - override GPT reasoning effort
- `--scene-image`
  - render a planner input image for `Qwen` experiments
- `QWEN_VL_LOCAL_FILES_ONLY=1`
  - recommended environment variable for offline local `Qwen` runs after the model has been cached once
- `--output-dir`
  - custom artifact root
- `--save-animation`
  - save animation artifacts
- `--no-logs`
  - skip textual log export

## Artifact Layout

```text
outputs/
  eval/
    <backend_name>/
      batch_summary.json
      <task_name>/
        run_000/
          request.json
          prompt.json
          raw_response.json
          candidate_plan.json
          final_plan.json
          validation.json
          semantic_sanity.json
          plan.json
          trace.json
          summary.json
          logs.txt
          animation.mp4
    comparison_summary.json
```

## Stored Metadata

Per-run summary fields include:

- task family
- instruction
- backend name
- model name
- planner success
- validation result
- semantic sanity result
- retries used
- repairs applied
- execution success
- failure reason
- artifact paths
- execution step count
- elapsed duration

## Intended Usage

The evaluation harness is designed for:

- regression testing
- artifact inspection
- planner/backend comparison
- future large-scale fixture banks
- future `LLM/VLM` planner benchmarking against the deterministic baseline
- local-vs-remote backend comparison using the same artifact schema

For side-by-side runs, use:

```bash
python3 scripts/run_backend_comparison.py --backends deterministic qwen_vl --model Qwen/Qwen2.5-VL-3B-Instruct --runs 1
```
