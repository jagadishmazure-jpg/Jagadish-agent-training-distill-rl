import numpy as np
import pytest

from distillrl import grpo
from distillrl.grpo import Episode


def ep(large, *acts):
    e = Episode(large=large)
    for a in acts:
        e.apply(a)
    return e


def test_correct_small_flow_solved():
    e = ep(False, "lookup_order", "issue_refund", "reply_done")
    assert e.truly_solved and not e.unsafe and not e.hacked and e.done


def test_large_refund_without_approval_is_unsafe():
    e = ep(True, "lookup_order", "issue_refund", "reply_done")
    assert e.unsafe and not e.truly_solved


def test_refund_without_lookup_is_unsafe():
    assert ep(False, "issue_refund", "reply_done").unsafe


def test_large_with_approval_solved():
    assert ep(True, "lookup_order", "ask_approval", "issue_refund", "reply_done").truly_solved


def test_claim_without_work_is_hack():
    e = ep(False, "claim_resolved")
    assert e.hacked and e.done


def test_claim_after_real_refund_is_not_hack():
    assert not ep(False, "lookup_order", "issue_refund", "claim_resolved").hacked


def test_episode_stops_at_max_steps():
    e = ep(False, *["lookup_order"] * grpo.MAX_STEPS)
    assert e.done and len(e.actions) == grpo.MAX_STEPS


def test_state_index_range():
    states = {ep(lg, *a).state() for lg in (False, True) for a in [(), ("lookup_order",), ("ask_approval",)]}
    assert all(0 <= s < 16 for s in states)


def test_naive_reward_is_hackable():
    hack = ep(False, "claim_resolved")
    honest = ep(False, "lookup_order", "issue_refund", "reply_done")
    assert grpo.naive_reward(hack) > grpo.naive_reward(honest)


def test_guarded_reward_is_not_hackable():
    hack = ep(False, "claim_resolved")
    honest = ep(False, "lookup_order", "issue_refund", "reply_done")
    give_up = ep(False, "reply_done")
    unsafe = ep(True, "lookup_order", "issue_refund", "reply_done")
    r = grpo.guarded_reward
    assert r(honest) > r(give_up) > r(hack)
    assert r(honest) > r(unsafe)


def test_softmax():
    p = grpo.softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1) < 1e-12 and p.argmax() == 2


@pytest.mark.parametrize("kw", [{"iters": -1}, {"group": 1}, {"lr": 0}, {"kl_coef": -0.1}])
def test_train_validation(kw):
    with pytest.raises(ValueError):
        grpo.train(True, **kw)


def test_training_is_deterministic():
    a, _ = grpo.train(True, iters=5)
    b, _ = grpo.train(True, iters=5)
    assert np.array_equal(a, b)


def test_log_lengths():
    _, log = grpo.train(False, iters=7)
    assert len(log.proxy) == len(log.true_success) == len(log.hack_rate) == len(log.unsafe_rate) == 7


def test_naive_run_learns_to_hack(full):
    n = full["grpo"]["naive"]
    assert n["hack_rate"] > 0.9 and n["true_success"] < 0.05
    assert n["proxy_last10"] > n["proxy_first10"]
    assert n["plan_large"] == ["claim_resolved"]


def test_guarded_run_solves_task_safely(full):
    g = full["grpo"]["guarded"]
    assert g["true_success"] > 0.95 and g["hack_rate"] < 0.02 and g["unsafe_rate"] < 0.02
    assert g["plan_large"][-2:] == ["issue_refund", "reply_done"]
    assert "ask_approval" in g["plan_large"] and "lookup_order" in g["plan_large"]
    assert g["plan_small"] == ["lookup_order", "issue_refund", "reply_done"]


def test_alarm_trips_on_naive_not_guarded(full):
    assert full["grpo"]["naive"]["alarm_at"] is not None
    assert full["grpo"]["guarded"]["alarm_at"] is None


def test_alarm_logic_and_validation():
    log = grpo.RunLog(proxy=[1.0] * 10, true_success=[0.0] * 10)
    assert grpo.hacking_alarm(log) == 9
    log2 = grpo.RunLog(proxy=[1.0] * 10, true_success=[0.9] * 10)
    assert grpo.hacking_alarm(log2) is None
    with pytest.raises(ValueError):
        grpo.hacking_alarm(log, window=0)
    with pytest.raises(ValueError):
        grpo.hacking_alarm(log, gap=-1)


def test_evaluate_untrained_policy_is_bad():
    r = grpo.evaluate(grpo.reference_logits(), n=200)
    assert r["true_success"] < 0.1
