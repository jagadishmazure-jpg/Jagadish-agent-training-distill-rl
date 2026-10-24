"""Small, honest latency helper: median and p95 over repeated single-request calls."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence


def latency_us(fn: Callable[[object], object], inputs: Sequence[object], repeats: int = 3) -> dict[str, float]:
    if repeats < 1 or not inputs:
        raise ValueError("need at least one input and one repeat")
    samples = []
    for _ in range(repeats):
        for x in inputs:
            t0 = time.perf_counter_ns()
            fn(x)
            samples.append((time.perf_counter_ns() - t0) / 1000.0)
    samples.sort()
    return {
        "p50_us": round(statistics.median(samples), 1),
        "p95_us": round(samples[int(0.95 * (len(samples) - 1))], 1),
        "n": len(samples),
    }
