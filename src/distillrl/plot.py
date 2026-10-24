"""Write the GRPO reward curves as CSV and as a dependency-free SVG line chart."""

from __future__ import annotations

from pathlib import Path

from .grpo import RunLog

W, H, PAD = 640, 300, 40
SERIES = [
    ("naive proxy reward", "naive", "proxy", "#d62728"),
    ("naive real success", "naive", "true_success", "#ff9896"),
    ("guarded proxy reward", "guarded", "proxy", "#1f77b4"),
    ("guarded real success", "guarded", "true_success", "#aec7e8"),
]


def csv_text(logs: dict[str, RunLog]) -> str:
    lines = ["iter,run,proxy_reward,true_success,hack_rate,unsafe_rate"]
    for name, log in logs.items():
        for i, row in enumerate(zip(log.proxy, log.true_success, log.hack_rate, log.unsafe_rate, strict=True)):
            lines.append(f"{i},{name}," + ",".join(f"{v:.4f}" for v in row))
    return "\n".join(lines) + "\n"


def svg_text(logs: dict[str, RunLog], lo: float = -1.5, hi: float = 1.2) -> str:
    n = max(len(log.proxy) for log in logs.values())

    def xy(i: int, v: float) -> str:
        x = PAD + (W - 2 * PAD) * i / max(n - 1, 1)
        y = H - PAD - (H - 2 * PAD) * (min(max(v, lo), hi) - lo) / (hi - lo)
        return f"{x:.1f},{y:.1f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" stroke="#888"/>',
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}" stroke="#888"/>',
        f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle">training iteration (0-{n - 1})</text>',
        f'<text x="{PAD}" y="{PAD - 10}">GRPO reward curves (y from {lo} to {hi})</text>',
    ]
    zero = xy(0, 0.0).split(",")[1]
    parts.append(f'<line x1="{PAD}" y1="{zero}" x2="{W - PAD}" y2="{zero}" stroke="#ddd" stroke-dasharray="4"/>')
    for k, (label, run, attr, colour) in enumerate(SERIES):
        vals = getattr(logs[run], attr)
        pts = " ".join(xy(i, v) for i, v in enumerate(vals))
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{pts}"/>')
        parts.append(f'<text x="{W - PAD - 150}" y="{PAD + 14 * k + 10}" fill="{colour}">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write(logs: dict[str, RunLog], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reward_curve.csv").write_text(csv_text(logs))
    (out_dir / "reward_curve.svg").write_text(svg_text(logs))
