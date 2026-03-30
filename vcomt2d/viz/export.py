"""Animation export helpers with MP4 support and graceful fallback guidance."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vcomt2d")

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, PillowWriter

from .animator import build_animation


class AnimationExportError(RuntimeError):
    pass


@dataclass
class ExportArtifact:
    saved_path: str
    format: str
    message: str


def resolve_ffmpeg_binary() -> Optional[str]:
    explicit = os.environ.get("VCOMT2D_FFMPEG_BINARY")
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            matplotlib.rcParams["animation.ffmpeg_path"] = str(explicit_path)
            return str(explicit_path)
    discovered = shutil.which("ffmpeg")
    if discovered:
        matplotlib.rcParams["animation.ffmpeg_path"] = discovered
    return discovered


def save_animation_mp4(trace, output_path: str, fps: int = 4) -> str:
    ffmpeg_binary = resolve_ffmpeg_binary()
    if ffmpeg_binary is None:
        raise AnimationExportError("ffmpeg is not installed. Install ffmpeg to enable MP4 export, or use GIF fallback.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation = build_animation(trace, fps=fps)
    writer = FFMpegWriter(fps=fps)
    animation.save(str(output), writer=writer)
    return str(output)


def save_animation_with_fallback(trace, output_stem: str, fps: int = 4, allow_gif_fallback: bool = True) -> ExportArtifact:
    mp4_path = f"{output_stem}.mp4"
    try:
        saved = save_animation_mp4(trace, mp4_path, fps=fps)
        return ExportArtifact(saved_path=saved, format="mp4", message="Saved MP4 animation.")
    except AnimationExportError as exc:
        if not allow_gif_fallback:
            raise
        gif_path = Path(f"{output_stem}.gif")
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        animation = build_animation(trace, fps=fps)
        animation.save(str(gif_path), writer=PillowWriter(fps=fps))
        return ExportArtifact(saved_path=str(gif_path), format="gif", message=f"{exc} GIF fallback saved instead.")
