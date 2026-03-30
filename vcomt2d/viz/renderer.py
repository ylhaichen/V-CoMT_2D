"""Lightweight matplotlib renderer for execution traces."""

from __future__ import annotations

import os
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vcomt2d")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


ROBOT_COLORS = {"robot_a": "#1f77b4", "robot_b": "#d62728"}
OBJECT_COLORS = {"ball": "#17becf", "box": "#7f7f7f"}


def _draw_world_snapshot(ax, snapshot, title: str, subtitle: str | None = None, robot_actions: Dict[str, dict] | None = None, events: list[str] | None = None) -> None:
    robot_actions = robot_actions or {}
    events = events or []
    bounds = snapshot["bounds"]
    ax.clear()
    ax.set_xlim(0, bounds[0])
    ax.set_ylim(0, bounds[1])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(subtitle or "")
    ax.grid(alpha=0.15)

    for goal in snapshot.get("goals", []):
        circle = Circle(goal["center"], goal["radius"], fill=False, linestyle="--", edgecolor="#2ca02c", linewidth=1.5)
        ax.add_patch(circle)
        ax.text(goal["center"][0], goal["center"][1] + goal["radius"] + 0.15, goal["region_id"], fontsize=8, ha="center")

    for door in snapshot.get("doors", []):
        color = "#2ca02c" if door["passable"] else "#8c564b"
        rect = Rectangle((door["pose"]["x"] - 0.15, door["pose"]["y"] - 0.8), 0.3, 1.6, color=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(door["pose"]["x"], door["pose"]["y"] + 1.0, door["door_id"], fontsize=8, ha="center")

    for obstacle in snapshot.get("obstacles", []):
        circle = Circle(obstacle["center"], obstacle["radius"], color="#444444", alpha=0.35)
        ax.add_patch(circle)

    for obj in snapshot.get("objects", []):
        color = OBJECT_COLORS.get(obj["object_type"], "#9467bd")
        circle = Circle((obj["pose"]["x"], obj["pose"]["y"]), obj.get("radius", 0.35), color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(obj["pose"]["x"], obj["pose"]["y"] + 0.4, obj["object_id"], fontsize=8, ha="center")

    legend_rows = []
    for robot in snapshot.get("robots", []):
        color = ROBOT_COLORS.get(robot["robot_id"], "#000000")
        circle = Circle((robot["pose"]["x"], robot["pose"]["y"]), 0.35, color=color, alpha=0.85)
        ax.add_patch(circle)
        action = robot_actions.get(robot["robot_id"], {})
        ax.text(robot["pose"]["x"], robot["pose"]["y"] + 0.45, robot["robot_id"], fontsize=8, ha="center")
        legend_rows.append(f"{robot['robot_id']}: {action.get('skill', 'IDLE')}")

    if events:
        legend_rows.extend(events[-2:])
    if legend_rows:
        ax.text(
            0.02,
            0.98,
            "\n".join(legend_rows),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )


def render_trace_frame(ax, frame, task_type: str) -> None:
    snapshot = frame.world_snapshot["world_state"]
    _draw_world_snapshot(
        ax,
        snapshot,
        title=f"{task_type} | tick={frame.tick} | state={frame.state_id}",
        subtitle=f"transition={frame.transition_reason} -> {frame.next_state}",
        robot_actions=frame.robot_actions,
        events=frame.events,
    )


def save_world_state_snapshot(world_state, output_path: str, title: str = "Planner Input Snapshot") -> str:
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_world_snapshot(ax, world_state.to_dict(), title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
