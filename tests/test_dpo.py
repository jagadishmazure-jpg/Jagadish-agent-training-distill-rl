import numpy as np
import pytest

from distillrl import dpo


def test_phi_shape_and_blocks():
    x = np.array([1, 0, 0, 1, 1.0])
    f = dpo.phi(x)
    assert f.shape == (len(dpo.REPLIES), len(dpo.CTX) * len(dpo.REPLIES))
    assert f.sum() == x.sum() * len(dpo.REPLIES)


def test_logprobs_normalised():
    lp = dpo.logprobs(dpo.reference_theta(), np.array([1, 0, 0, 0, 1.0]))
    assert abs(np.exp(lp).sum() - 1) < 1e-12


@pytest.mark.parametrize(
    "x,reply",
    [
        ([0, 0, 1, 0, 1], "refuse"),
        ([1, 0, 0, 1, 1], "hand_off"),
        ([1, 0, 0, 0, 1], "ask_confirm"),
        ([1, 1, 0, 0, 1], "act_now"),
        ([0, 0, 0, 0, 1], "act_now"),
    ],
)
def test_house_rule(x, reply):
    assert dpo.REPLIES[dpo.preferred(np.array(x, dtype=float))] == reply


def test_reference_prefers_acting():
    x = np.array([1, 0, 0, 0, 1.0])
    assert dpo.REPLIES[int(np.argmax(dpo.logprobs(dpo.reference_theta(), x)))] == "act_now"


def test_pairs_have_distinct_replies_and_noise():
    clean = dpo.make_pairs(300, 1, noise=0.0)
    assert all(p.chosen != p.rejected for p in clean)
    assert all(p.chosen == dpo.preferred(p.x) for p in clean)
    noisy = dpo.make_pairs(300, 1, noise=0.5)
    assert any(p.chosen != dpo.preferred(p.x) for p in noisy)


@pytest.mark.parametrize("kw", [{"n": -1, "seed": 0}, {"n": 5, "seed": 0, "noise": 1.5}])
def test_make_pairs_validation(kw):
    with pytest.raises(ValueError):
        dpo.make_pairs(**kw)


@pytest.mark.parametrize("kw", [{"beta": 0}, {"lr": -1}, {"epochs": -1}])
def test_train_validation(kw):
    with pytest.raises(ValueError):
        dpo.dpo_train(dpo.make_pairs(5, 0), **kw)


def test_zero_epochs_returns_reference():
    theta, losses = dpo.dpo_train(dpo.make_pairs(5, 0), epochs=0)
    assert np.array_equal(theta, dpo.reference_theta()) and losses == []


def test_analytic_gradient_matches_numeric():
    """The hand-written DPO gradient agrees with finite differences."""
    pairs = dpo.make_pairs(20, 5)
    rng = np.random.default_rng(0)
    theta = dpo.reference_theta() + rng.normal(0, 0.3, size=dpo.reference_theta().shape)
    ref, beta = dpo.reference_theta(), 1.0

    def loss(th):
        total = 0.0
        for p in pairs:
            lp, lr = dpo.logprobs(th, p.x), dpo.logprobs(ref, p.x)
            m = beta * ((lp[p.chosen] - lr[p.chosen]) - (lp[p.rejected] - lr[p.rejected]))
            total += np.log1p(np.exp(-m))
        return total / len(pairs)

    grad = np.zeros_like(theta)
    for p in pairs:
        f = dpo.phi(p.x)
        lp, lr = dpo.logprobs(theta, p.x), dpo.logprobs(ref, p.x)
        m = beta * ((lp[p.chosen] - lr[p.chosen]) - (lp[p.rejected] - lr[p.rejected]))
        s = 1 / (1 + np.exp(-m))
        grad += -(1 - s) * beta * (f[p.chosen] - f[p.rejected])
    grad /= len(pairs)
    eps = 1e-6
    for i in range(0, len(theta), 3):
        d = np.zeros_like(theta)
        d[i] = eps
        num = (loss(theta + d) - loss(theta - d)) / (2 * eps)
        assert abs(num - grad[i]) < 1e-6


def test_dpo_improves_and_stays_bounded(full):
    p = full["dpo"]
    assert p["pair_acc_after"] >= 0.95 > p["pair_acc_before"]
    assert p["rule_agree_after"] >= 0.95
    assert p["loss_last"] < p["loss_first"]
    assert 0 < p["kl_to_ref"] < 2.0


def test_mean_kl_zero_at_reference():
    assert abs(dpo.mean_kl(dpo.reference_theta(), n=50)) < 1e-12
