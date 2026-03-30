"""Shared geometric and serialization-friendly types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class Pose2D:
    """Serializable 2D pose."""

    x: float
    y: float
    theta: float = 0.0

    def xy(self) -> Vec2:
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Pose2D":
        return Pose2D(x=float(data["x"]), y=float(data["y"]), theta=float(data.get("theta", 0.0)))


def distance_xy(a: Vec2, b: Vec2) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def distance_pose(a: Pose2D, b: Pose2D) -> float:
    return distance_xy(a.xy(), b.xy())


def midpoint(a: Vec2, b: Vec2) -> Vec2:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def to_serializable(value: Any) -> Any:
    """Convert nested dataclasses and tuples to JSON-friendly values."""

    if is_dataclass(value):
        return {key: to_serializable(inner) for key, inner in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_serializable(item) for item in value)
    return value


def nearest_by_distance(origin: Pose2D, candidates: Iterable[Tuple[str, Pose2D]]) -> str:
    ordered = sorted(candidates, key=lambda item: (distance_pose(origin, item[1]), item[0]))
    if not ordered:
        raise ValueError("No candidate poses provided")
    return ordered[0][0]

