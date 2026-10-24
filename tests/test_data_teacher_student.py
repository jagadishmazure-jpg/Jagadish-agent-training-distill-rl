import pytest

from distillrl import data, student, teacher
from distillrl.data import TOOLS, Example


def test_dataset_is_deterministic():
    assert data.make_dataset(50, 1) == data.make_dataset(50, 1)
    assert data.make_dataset(50, 1) != data.make_dataset(50, 2)


def test_negative_dataset_size_rejected():
    with pytest.raises(ValueError):
        data.make_dataset(-1, 0)


def test_split_sizes():
    s, p, t = data.splits()
    assert (len(s), len(p), len(t)) == (data.SEED_SIZE, data.POOL_SIZE, data.TEST_SIZE)


def test_every_tool_appears_in_each_split():
    for part in data.splits():
        assert {e.tool for e in part} == set(TOOLS)


def test_every_tool_has_templates():
    assert set(data.TEMPLATES) == set(TOOLS)


@pytest.mark.parametrize(
    "text,tool",
    [
        ("please refund BW-11111", "issue_refund"),
        ("where is my package", "track_shipment"),
        ("cancel BW-1", "cancel_order"),
        ("let me talk to a real person", "escalate_human"),
        ("what is your return policy", "answer_faq"),
        ("change the delivery address", "update_address"),
        ("can you pull up order BW-2", "lookup_order"),
    ],
)
def test_rules_on_clear_cases(text, tool):
    assert teacher.rule_label(text) == tool


def test_rules_negation_lowers_refund():
    assert teacher.scores("no refund needed")["issue_refund"] < 0


def test_rules_unmatched_goes_to_human():
    assert teacher.rule_label("zzz qqq") == "escalate_human"


def test_rule_distribution_sums_to_one():
    d = teacher.rule_distribution("please refund my tent")
    assert abs(sum(d.values()) - 1) < 1e-9 and max(d, key=d.get) == "issue_refund"


def test_teacher_requires_all_tools():
    with pytest.raises(ValueError):
        teacher.fit_teacher([Example("refund", "issue_refund")])


def test_teacher_distributions_are_probabilities(models):
    t, _ = models
    d = t.distributions(["please refund BW-1", "where is my parcel"])
    assert d.shape == (2, len(TOOLS))
    assert abs(d.sum(axis=1) - 1).max() < 1e-6


def test_student_uses_teacher_labels_not_gold(models):
    t, _ = models
    pool = [Example("please refund BW-1", "answer_faq")]  # wrong gold on purpose
    _, labels, _ = student.teacher_labels(t, pool)
    assert labels == ["issue_refund"]


def test_student_confidence_floor(models):
    t, _ = models
    _, _, w = student.teacher_labels(t, data.splits()[1][:50])
    assert w.min() >= 0.2 and w.max() <= 1.0


def test_student_predicts_known_tools(models):
    _, s = models
    assert set(s.predict(["refund me", "cancel it", "human please"])) <= set(TOOLS)


def test_student_is_much_smaller_than_teacher(models):
    t, s = models
    assert s.size_bytes() * 20 < t.size_bytes()


def test_accuracy_helper_validates():
    assert student.accuracy(["a", "b"], ["a", "c"]) == 0.5
    with pytest.raises(ValueError):
        student.accuracy([], [])
    with pytest.raises(ValueError):
        student.accuracy(["a"], ["a", "b"])


def test_distill_report_thresholds(full):
    d = full["distill"]
    assert d["teacher_acc"] >= 0.98
    assert d["student_acc"] >= d["teacher_acc"] - 0.01
    assert d["student_teacher_agreement"] >= 0.98
    assert d["rules_only_acc"] < d["teacher_acc"]


def test_off_template_set_is_a_known_weakness(full):
    """Kept visible on purpose: both models drop to roughly half on unseen phrasings."""
    d = full["distill"]
    assert d["hard_n"] == 20
    assert d["hard_student_acc"] <= 0.7 and d["hard_teacher_acc"] <= 0.7


def test_hard_set_labels_valid():
    from distillrl.hardset import HARD

    assert {e.tool for e in HARD} == set(TOOLS)
