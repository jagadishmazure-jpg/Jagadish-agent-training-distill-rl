"""Teacher: the big, slow labeller whose behaviour we distil.

Two parts, averaged:
1. A hand-written weighted keyword scorer (~60 regex rules, softmax over rule scores).
2. A large character 1-5-gram logistic model (2^18 hashed features, about 14 MB of weights)
   fitted on a small gold-labelled seed set of 500 messages.

In a production stack the teacher would be a large hosted model. Here it is local and free,
but it has the same shape: bigger and slower than what you want to serve.
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from .data import TOOLS, Example

RULES: dict[str, list[tuple[str, float]]] = {
    "lookup_order": [
        (r"\bpull up\b", 2.0),
        (r"\bdetails?\b", 1.5),
        (r"\breceipt\b", 2.0),
        (r"\bwhat is in\b", 2.0),
        (r"\bgo through\b", 1.5),
        (r"\bwhat did i pay\b", 2.0),
        (r"\bwhat i bought\b", 2.0),
        (r"\border\b", 0.4),
    ],
    "track_shipment": [
        (r"\bwhere is\b", 2.0),
        (r"\bshipped\b", 2.0),
        (r"\btracking\b", 2.5),
        (r"\barrive\b", 2.0),
        (r"\blate\b", 1.5),
        (r"\bpackage\b", 1.0),
        (r"\bparcel\b", 1.0),
        (r"\bcourier\b", 1.5),
        (r"\bshown up\b", 2.0),
        (r"\bshipping\b", 0.6),
    ],
    "issue_refund": [
        (r"\brefund\b", 2.5),
        (r"\bmoney back\b", 2.5),
        (r"\breversed?\b", 2.0),
        (r"\bcharged twice\b", 2.5),
        (r"\bbroken\b", 1.0),
        (r"\bfaulty\b", 1.0),
        (r"\bcredit my\b", 2.0),
        (r"\breturn the\b", 1.0),
        (r"\bno refund\b", -3.0),
    ],
    "update_address": [
        (r"\baddress\b", 2.5),
        (r"\bmoved\b", 1.5),
        (r"\bnew place\b", 2.0),
        (r"\bwrong street\b", 2.5),
        (r"\boffice\b", 1.5),
        (r"\binstead\b", 0.5),
    ],
    "cancel_order": [
        (r"\bcancel\b", 2.5),
        (r"\bcall off\b", 2.5),
        (r"\bscrap\b", 2.0),
        (r"\bkill the order\b", 2.5),
        (r"\bdo not want\b", 2.0),
        (r"\bby mistake\b", 1.0),
        (r"\bwrong\b", 0.3),
    ],
    "escalate_human": [
        (r"\breal person\b", 3.0),
        (r"\bmanager\b", 2.5),
        (r"\bspeak to\b", 2.0),
        (r"\bhuman\b", 2.5),
        (r"\bdispute\b", 2.0),
        (r"\bagent\b", 2.0),
        (r"\bthird time\b", 1.5),
        (r"\buseless\b", 1.0),
    ],
    "answer_faq": [
        (r"\bpolicy\b", 2.5),
        (r"\bship to\b", 2.0),
        (r"\bwarranty\b", 2.5),
        (r"\bopen on\b", 2.5),
        (r"\bsizes?\b", 2.0),
        (r"\bwash\b", 2.5),
        (r"\bprice match\b", 3.0),
        (r"\bdo you\b", 0.8),
        (r"\bhow do i\b", 0.8),
    ],
}
COMPILED = {t: [(re.compile(p), w) for p, w in rs] for t, rs in RULES.items()}
TEMPERATURE = 1.0


def scores(text: str) -> dict[str, float]:
    low = text.lower()
    return {t: sum(w for rx, w in COMPILED[t] if rx.search(low)) for t in TOOLS}


def rule_distribution(text: str) -> dict[str, float]:
    s = scores(text)
    m = max(s.values())
    exp = {t: math.exp((v - m) / TEMPERATURE) for t, v in s.items()}
    z = sum(exp.values())
    return {t: e / z for t, e in exp.items()}


def rule_label(text: str) -> str:
    s = scores(text)
    best = max(s.values())
    if best <= 0:
        return "escalate_human"  # nothing matched: hand to a person rather than guess
    return next(t for t in TOOLS if s[t] == best)


def rule_count() -> int:
    return sum(len(v) for v in RULES.values())


BIG_FEATURES = 2**18


@dataclass
class Teacher:
    vec: HashingVectorizer
    clf: LogisticRegression
    rule_weight: float = 0.5

    def distributions(self, texts: list[str]) -> np.ndarray:
        model = self.clf.predict_proba(self.vec.transform(texts))
        order = [list(self.clf.classes_).index(t) for t in TOOLS]
        model = model[:, order]
        rules = np.array([[rule_distribution(t)[k] for k in TOOLS] for t in texts])
        return self.rule_weight * rules + (1 - self.rule_weight) * model

    def label_many(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        d = self.distributions(texts)
        return [TOOLS[i] for i in d.argmax(axis=1)], d.max(axis=1)

    def label(self, text: str) -> str:
        return self.label_many([text])[0][0]

    def size_bytes(self) -> int:
        return len(pickle.dumps(self.clf, protocol=5)) + len(pickle.dumps(RULES, protocol=5))


def fit_teacher(seed_set: list[Example], seed: int = 0) -> Teacher:
    if len({e.tool for e in seed_set}) != len(TOOLS):
        raise ValueError("seed set must cover every tool")
    vec = HashingVectorizer(analyzer="char_wb", ngram_range=(1, 5), n_features=BIG_FEATURES, alternate_sign=False)
    clf = LogisticRegression(C=10.0, max_iter=1000, random_state=seed)
    clf.fit(vec.transform([e.text for e in seed_set]), [e.tool for e in seed_set])
    return Teacher(vec, clf)
