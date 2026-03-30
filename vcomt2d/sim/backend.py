"""Future-facing backend wrapper for the lightweight 2D runtime."""

from __future__ import annotations

from dataclasses import dataclass

from .world import SimWorld
from .entities import WorldState


@dataclass
class SimBackend:
    """Thin compatibility layer for future simulator backends."""

    def create_world(self, world_state: WorldState) -> SimWorld:
        return SimWorld(state=world_state)

