"""distillrl demo | bench | curves"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plot
from .report import full

ROOT = Path.cwd()


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def tables(r: dict | None = None) -> dict[str, str]:
    r = r or full()
    d, p, g = r["distill"], r["dpo"], r["grpo"]
    distill = "\n".join(
        [
            "| model | gold test accuracy (n=1,000) | off-template accuracy (n=20) | size |",
            "| --- | --- | --- | --- |",
            f"| rules only ({d['rule_count']} regex rules) | {pct(d['rules_only_acc'])} | - | - |",
            f"| teacher (rules + 2^18-feature char model) | {pct(d['teacher_acc'])} | {pct(d['hard_teacher_acc'])} "
            f"| {d['teacher_bytes'] / 1e6:.2f} MB |",
            f"| student (2^12-feature linear, teacher labels only) | {pct(d['student_acc'])} "
            f"| {pct(d['hard_student_acc'])} | {d['student_bytes'] / 1e3:.0f} kB |",
            "",
            f"Student agrees with the teacher on {pct(d['student_teacher_agreement'])} of the test set and is "
            f"{d['teacher_bytes'] / d['student_bytes']:.0f}x smaller.",
        ]
    )
    dpo_t = "\n".join(
        [
            "| metric | reference (start) | after DPO |",
            "| --- | --- | --- |",
            f"| held-out pair accuracy (n={p['test_pairs']}) | {pct(p['pair_acc_before'])} | {pct(p['pair_acc_after'])} |",
            f"| top reply matches hidden house rule (n=2,000) | {pct(p['rule_agree_before'])} | {pct(p['rule_agree_after'])} |",
            f"| DPO loss | {p['loss_first']:.3f} | {p['loss_last']:.3f} |",
            f"| mean KL to reference | 0.000 | {p['kl_to_ref']:.3f} |",
        ]
    )
    rows = [
        "| policy | own training reward, first 10 iters | own training reward, last 10 iters | real success | hack rate "
        "| unsafe refunds | greedy plan (large refund) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| untrained | - | - | {pct(g['start']['true_success'])} | {pct(g['start']['hack_rate'])} "
        f"| {pct(g['start']['unsafe_rate'])} | random |",
    ]
    for name in ("naive", "guarded"):
        x = g[name]
        rows.append(
            f"| {name} reward | {x['proxy_first10']:.3f} | {x['proxy_last10']:.3f} | {pct(x['true_success'])} "
            f"| {pct(x['hack_rate'])} | {pct(x['unsafe_rate'])} | {' -> '.join(x['plan_large'])} |"
        )
    alarm = g["naive"]["alarm_at"]
    rows += [
        "",
        f"Hacking alarm (proxy minus real success > 0.5 over 10 iterations): naive run trips at iteration "
        f"{alarm}; guarded run: {'never' if g['guarded']['alarm_at'] is None else g['guarded']['alarm_at']}.",
    ]
    return {"distill": distill, "dpo": dpo_t, "grpo": "\n".join(rows)}


def bench_tables(b: dict) -> str:
    q, dl, m = b["quant"], b["distill_latency"], b["machine"]
    return "\n".join(
        [
            "| model | accuracy (n=1,000) | serialized size | p50 latency | p95 latency |",
            "| --- | --- | --- | --- | --- |",
            f"| teacher | see above | see above | {dl['teacher']['p50_us']:.0f} us | {dl['teacher']['p95_us']:.0f} us |",
            f"| student | see above | see above | {dl['student']['p50_us']:.0f} us | {dl['student']['p95_us']:.0f} us |",
            f"| MLP fp32 | {pct(q['fp32_acc'])} | {q['fp32_bytes'] / 1e6:.2f} MB | {q['fp32_latency']['p50_us']:.0f} us "
            f"| {q['fp32_latency']['p95_us']:.0f} us |",
            f"| MLP int8 (dynamic) | {pct(q['int8_acc'])} | {q['int8_bytes'] / 1e6:.2f} MB "
            f"| {q['int8_latency']['p50_us']:.0f} us | {q['int8_latency']['p95_us']:.0f} us |",
            "",
            f"Recorded on the build box: Python {m['python']}, torch {m['torch']}, {m['processor']}, "
            f"{m['cpu_threads_used']} thread, batch size 1, {q['fp32_latency']['n']} timed calls per row. "
            f"Largest fp32-vs-int8 logit gap: {q['max_logit_gap']}.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="distillrl")
    ap.add_argument("cmd", choices=["demo", "bench", "curves"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    if a.cmd == "bench":
        from . import bench

        print(bench_tables(bench.write(out / "bench.json")))
        return 0
    r = full()
    if a.cmd == "curves":
        plot.write({k: r["grpo"][k]["curve"] for k in ("naive", "guarded")}, out)
        print(f"wrote {out / 'reward_curve.csv'} and {out / 'reward_curve.svg'}")
        return 0
    for name, t in tables(r).items():
        print(f"## {name}\n\n{t}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
