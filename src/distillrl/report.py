"""One deterministic run of every experiment. Used by the CLI, the docs renderer and the tests.

Accuracy, size and reward numbers here are deterministic for a given dependency set.
Latency is not, so it lives in `bench.py` and is recorded to results/bench.json.
"""

from __future__ import annotations

from functools import lru_cache

from . import dpo, grpo, student, teacher
from .data import POOL_SIZE, SEED_SIZE, TEST_SIZE, splits
from .hardset import HARD


@lru_cache(maxsize=1)
def models() -> tuple[teacher.Teacher, student.Student]:
    seed_set, pool, _ = splits()
    t = teacher.fit_teacher(seed_set)
    return t, student.train(t, pool)


def distill() -> dict[str, float | int]:
    _, _, test = splits()
    t, s = models()
    texts, gold = [e.text for e in test], [e.tool for e in test]
    t_pred, s_pred = t.label_many(texts)[0], s.predict(texts)
    h_texts, h_gold = [e.text for e in HARD], [e.tool for e in HARD]
    return {
        "seed_gold": SEED_SIZE,
        "pool_unlabelled": POOL_SIZE,
        "test_gold": TEST_SIZE,
        "rules_only_acc": round(student.accuracy([teacher.rule_label(x) for x in texts], gold), 3),
        "teacher_acc": round(student.accuracy(t_pred, gold), 3),
        "student_acc": round(student.accuracy(s_pred, gold), 3),
        "student_teacher_agreement": round(student.accuracy(s_pred, t_pred), 3),
        "teacher_bytes": t.size_bytes(),
        "student_bytes": s.size_bytes(),
        "hard_n": len(HARD),
        "hard_teacher_acc": round(student.accuracy(t.label_many(h_texts)[0], h_gold), 3),
        "hard_student_acc": round(student.accuracy(s.predict(h_texts), h_gold), 3),
        "rule_count": teacher.rule_count(),
    }


@lru_cache(maxsize=1)
def full() -> dict[str, object]:
    return {"distill": distill(), "dpo": dpo.run(), "grpo": grpo.run()}
