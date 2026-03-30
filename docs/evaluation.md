# Evaluation

## Purpose

The evaluation harness turns the planner/executor stack into a benchmark-ready subsystem.

It supports:

- deterministic batch runs
- GPT-backed batch runs
- per-run artifacts
- machine-readable summaries
- animation export during evaluation
- failure analysis through stored traces and logs

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
  - choose `deterministic`, `gpt`, `llm_stub`, or `vlm_stub`
- `--model`
  - override the GPT model name
- `--reasoning-effort`
  - override GPT reasoning effort
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

For side-by-side runs, use:

```bash
python3 scripts/run_backend_comparison.py --model gpt-5.4 --runs 1
```
