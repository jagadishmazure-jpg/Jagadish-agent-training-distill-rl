import json
from pathlib import Path

import pytest
import torch

from distillrl import cli, data, plot, quant, timing
from distillrl.grpo import RunLog

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small_models(models):
    t, _ = models
    pool = data.splits()[1][:800]
    fp32 = quant.train_fp32(t, pool, epochs=3)
    return fp32, quant.to_int8(fp32)


def test_int8_is_smaller(small_models):
    fp32, int8 = small_models
    assert quant.size_bytes(int8) * 3 < quant.size_bytes(fp32)


def test_int8_keeps_accuracy(small_models):
    fp32, int8 = small_models
    test = data.splits()[2][:300]
    x, gold = quant.features([e.text for e in test]), [e.tool for e in test]
    from distillrl.student import accuracy

    a32, a8 = accuracy(quant.predict(fp32, x), gold), accuracy(quant.predict(int8, x), gold)
    assert a32 > 0.9 and abs(a32 - a8) <= 0.02


def test_int8_layers_are_quantized(small_models):
    _, int8 = small_models
    kinds = {type(m).__module__ for m in int8.modules()}
    assert any("quantized" in k for k in kinds)


def test_logit_gap_small(small_models):
    fp32, int8 = small_models
    x = quant.features(["please refund BW-1", "where is my parcel"])
    assert quant.max_logit_gap(fp32, int8, x) < 1.0


def test_negative_epochs_rejected(models):
    with pytest.raises(ValueError):
        quant.train_fp32(models[0], [], epochs=-1)


def test_features_shape():
    x = quant.features(["a", "b"])
    assert x.dtype == torch.float32 and x.shape[0] == 2


def test_latency_helper():
    r = timing.latency_us(lambda v: v, [1, 2, 3], repeats=2)
    assert r["n"] == 6 and r["p50_us"] >= 0 and r["p95_us"] >= r["p50_us"]
    with pytest.raises(ValueError):
        timing.latency_us(lambda v: v, [], repeats=1)
    with pytest.raises(ValueError):
        timing.latency_us(lambda v: v, [1], repeats=0)


def test_recorded_bench_is_sane():
    b = json.loads((ROOT / "results" / "bench.json").read_text())
    q = b["quant"]
    assert q["int8_bytes"] < q["fp32_bytes"]
    assert abs(q["fp32_acc"] - q["int8_acc"]) <= 0.02
    assert b["distill_latency"]["student"]["p50_us"] < b["distill_latency"]["teacher"]["p50_us"]
    assert b["machine"]["cpu_threads_used"] == 1


def test_bench_validation():
    from distillrl import bench

    with pytest.raises(ValueError):
        bench.run(n_latency=0)


def test_csv_and_svg():
    logs = {
        "naive": RunLog([0.1, 0.5], [0.0, 0.0], [0.5, 1.0], [0.0, 0.0]),
        "guarded": RunLog([-1, 1], [0, 1], [0, 0], [0, 0]),
    }
    csv = plot.csv_text(logs)
    assert csv.splitlines()[0].startswith("iter,run") and len(csv.splitlines()) == 5
    svg = plot.svg_text(logs)
    assert svg.startswith("<svg") and svg.count("<polyline") == 4


def test_committed_curve_matches_fresh_run(full):
    logs = {k: full["grpo"][k]["curve"] for k in ("naive", "guarded")}
    assert (ROOT / "results" / "reward_curve.csv").read_text() == plot.csv_text(logs)


def test_tables_render(full):
    t = cli.tables(full)
    assert set(t) == {"distill", "dpo", "grpo"}
    assert "claim_resolved" in t["grpo"] and "64x smaller" in t["distill"]


def test_bench_table_from_recorded_file():
    b = json.loads((ROOT / "results" / "bench.json").read_text())
    assert "MLP int8" in cli.bench_tables(b)


def test_cli_demo_runs(capsys):
    assert cli.main(["demo"]) == 0
    assert "## grpo" in capsys.readouterr().out


def test_cli_rejects_unknown():
    with pytest.raises(SystemExit):
        cli.main(["nope"])
