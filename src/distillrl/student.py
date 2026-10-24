"""Student: a small linear model distilled from the teacher's labels.

The student never sees gold labels. It is trained only on what the teacher said about an
unlabelled pool, weighted by the teacher's confidence, and then compared with the teacher on
a separate gold test set.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from . import teacher
from .data import TOOLS, Example

N_FEATURES = 2**12


def vectorizer() -> HashingVectorizer:
    return HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=N_FEATURES, alternate_sign=False, norm="l2")


@dataclass
class Student:
    vec: HashingVectorizer
    clf: LogisticRegression

    def predict(self, texts: list[str]) -> list[str]:
        return [str(t) for t in self.clf.predict(self.vec.transform(texts))]

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    def size_bytes(self) -> int:
        """Size of what you would ship: the learned weights (the vectoriser is stateless)."""
        return len(pickle.dumps(self.clf, protocol=5))


def teacher_labels(t: teacher.Teacher, pool: list[Example]) -> tuple[list[str], list[str], np.ndarray]:
    """Gold labels in `pool` are ignored: only the teacher's opinion is used."""
    texts = [e.text for e in pool]
    labels, conf = t.label_many(texts)
    return texts, labels, np.maximum(conf, 0.2)


def train(t: teacher.Teacher, pool: list[Example], *, use_confidence: bool = True, seed: int = 0) -> Student:
    texts, labels, weights = teacher_labels(t, pool)
    vec = vectorizer()
    clf = LogisticRegression(C=4.0, max_iter=400, random_state=seed)
    clf.fit(vec.transform(texts), labels, sample_weight=weights if use_confidence else None)
    missing = set(TOOLS) - set(clf.classes_)
    if missing:
        raise ValueError(f"teacher never produced labels for: {sorted(missing)}")
    return Student(vec, clf)


def accuracy(pred: list[str], gold: list[str]) -> float:
    if len(pred) != len(gold) or not gold:
        raise ValueError("prediction and gold lists must be the same non-zero length")
    return sum(p == g for p, g in zip(pred, gold, strict=True)) / len(gold)
