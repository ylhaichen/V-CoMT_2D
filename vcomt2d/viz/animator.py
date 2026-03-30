"""Matplotlib animation helpers for execution traces."""

from __future__ import annotations

import os
from typing import Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vcomt2d")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .renderer import render_trace_frame


def build_animation(trace, fps: int = 4, figsize: Tuple[int, int] = (7, 7)) -> FuncAnimation:
    fig, ax = plt.subplots(figsize=figsize)

    def update(frame_index: int):
        render_trace_frame(ax, trace.frames[frame_index], trace.task_type)
        return []

    animation = FuncAnimation(fig, update, frames=len(trace.frames), interval=1000 / fps, blit=False, repeat=False)
    return animation
