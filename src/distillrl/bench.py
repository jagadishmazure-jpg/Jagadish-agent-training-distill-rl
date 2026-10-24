"""Latency and int8-vs-fp32 serving benchmark. Machine-dependent, so results are recorded.

    distillrl bench            # writes results/bench.json
The README latency table is rendered from that file, together with the machine it ran on.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

from . import quant, student
from .data import splits
from .report import models
from .timing import latency_us


def run(n_latency: int = 300, repeats: int = 3) -> dict[str, object]:
    if n_latency < 1 or repeats < 1:
        raise ValueError("n_latency and repeats must be positive")
    torch.set_num_threads(1)
    _, pool, test = splits()
    t, s = models()
    texts, gold = [e.text for e in test], [e.tool for e in test]
    sample = texts[:n_latency]

    fp32 = quant.train_fp32(t, pool)
    int8 = quant.to_int8(fp32)
    x = quant.features(texts)
    rows = [x[i : i + 1] for i in range(n_latency)]
    with torch.no_grad():
        fp32_lat = latency_us(fp32, rows, repeats)
        int8_lat = latency_us(int8, rows, repeats)
    return {
        "machine": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "processor": platform.machine(),
            "cpu_threads_used": 1,
        },
        "distill_latency": {
            "teacher": latency_us(t.label, sample, repeats),
            "student": latency_us(s.predict_one, sample, repeats),
        },
        "quant": {
            "fp32_acc": round(student.accuracy(quant.predict(fp32, x), gold), 3),
            "int8_acc": round(student.accuracy(quant.predict(int8, x), gold), 3),
            "fp32_bytes": quant.size_bytes(fp32),
            "int8_bytes": quant.size_bytes(int8),
            "max_logit_gap": round(quant.max_logit_gap(fp32, int8, x), 4),
            "fp32_latency": fp32_lat,
            "int8_latency": int8_lat,
        },
    }


def write(path: Path) -> dict[str, object]:
    out = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out
