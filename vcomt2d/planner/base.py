"""Planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import PlanningRequest, PlanningResult


class Planner(ABC):
    @abstractmethod
    def plan(self, request: PlanningRequest) -> PlanningResult:
        raise NotImplementedError

