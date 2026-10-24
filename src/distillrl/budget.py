"""Training-run budget. Every cost and limit must be a finite, non-negative number."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .pii import SafeError


def _check(name: str, value: float, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SafeError(f"{name} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise SafeError(f"{name} must be finite")
    if value < 0:
        raise SafeError(f"{name} must not be negative (got {value})")
    if integer and int(value) != value:
        raise SafeError(f"{name} must be a whole number")


@dataclass(frozen=True)
class Budget:
    """Caps for one training run. CPU cost uses an amortised hourly rate for a laptop or small VM."""

    max_steps: int = 400
    max_cpu_seconds: float = 120.0
    cpu_cost_per_hour_usd: float = 0.0
    max_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        _check("max_steps", self.max_steps, integer=True)
        _check("max_cpu_seconds", self.max_cpu_seconds)
        _check("cpu_cost_per_hour_usd", self.cpu_cost_per_hour_usd)
        _check("max_cost_usd", self.max_cost_usd)

    def cost(self, cpu_seconds: float) -> float:
        _check("cpu_seconds", cpu_seconds)
        return round(self.cpu_cost_per_hour_usd * cpu_seconds / 3600.0, 6)

    def allows(self, steps: int, cpu_seconds: float) -> bool:
        _check("steps", steps, integer=True)
        _check("cpu_seconds", cpu_seconds)
        if steps > self.max_steps or cpu_seconds > self.max_cpu_seconds:
            return False
        return self.cpu_cost_per_hour_usd == 0 or self.cost(cpu_seconds) <= self.max_cost_usd
