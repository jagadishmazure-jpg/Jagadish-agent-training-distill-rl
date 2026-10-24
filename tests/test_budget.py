import math

import pytest

from distillrl.budget import Budget
from distillrl.pii import SafeError


def test_defaults_are_free():
    b = Budget()
    assert b.cost(3600) == 0.0 and b.allows(10, 1.0)


@pytest.mark.parametrize("field", ["max_steps", "max_cpu_seconds", "cpu_cost_per_hour_usd", "max_cost_usd"])
def test_negative_values_rejected(field):
    with pytest.raises(SafeError, match="negative"):
        Budget(**{field: -1})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected(bad):
    with pytest.raises(SafeError):
        Budget(max_cost_usd=bad)


@pytest.mark.parametrize("bad", ["5", None, True])
def test_non_numbers_rejected(bad):
    with pytest.raises(SafeError):
        Budget(max_cost_usd=bad)


def test_fractional_steps_rejected():
    with pytest.raises(SafeError):
        Budget(max_steps=1.5)


def test_negative_runtime_inputs_rejected():
    b = Budget()
    with pytest.raises(SafeError):
        b.cost(-1)
    with pytest.raises(SafeError):
        b.allows(-1, 1)
    with pytest.raises(SafeError):
        b.allows(1, -0.5)


def test_cost_cap_enforced():
    b = Budget(cpu_cost_per_hour_usd=0.36, max_cost_usd=0.01, max_cpu_seconds=1000)
    assert b.cost(100) == 0.01
    assert b.allows(1, 100)
    assert not b.allows(1, 101)


def test_step_and_time_caps():
    b = Budget(max_steps=5, max_cpu_seconds=2)
    assert not b.allows(6, 1)
    assert not b.allows(1, 3)
