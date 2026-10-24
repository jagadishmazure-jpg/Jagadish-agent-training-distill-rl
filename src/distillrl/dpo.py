"""Direct Preference Optimisation on a toy reply policy (numpy, no GPU).

Setting: the Brindlewood Gear support agent picks one of four reply styles for a situation.
A hidden house rule decides which reply people prefer; we only ever show the learner
pairs (chosen, rejected) generated from that rule with 10% label noise.

Policy: pi(y | x) = softmax_y(theta . phi(x, y)), with phi = situation features crossed with a
one-hot reply style. The reference policy is the starting point (an "SFT" model that prefers
acting immediately). DPO loss per pair, with beta controlling how far we may drift:

    L = -log sigmoid( beta * [(log pi(yw) - log ref(yw)) - (log pi(yl) - log ref(yl))] )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REPLIES = ("act_now", "ask_confirm", "refuse", "hand_off")
CTX = ("risky", "consented", "injection", "angry", "bias")  # "bias" is a constant 1 feature


def situation(rng: np.random.Generator) -> np.ndarray:
    risky = rng.random() < 0.5
    consented = risky and rng.random() < 0.4
    injection = rng.random() < 0.15
    angry = rng.random() < 0.25
    return np.array([risky, consented, injection, angry, 1.0], dtype=float)


def preferred(x: np.ndarray) -> int:
    """The hidden house rule the learner must recover from pairs alone."""
    risky, consented, injection, angry, _ = x.astype(bool)
    if injection:
        return REPLIES.index("refuse")
    if angry and risky and not consented:
        return REPLIES.index("hand_off")
    if risky and not consented:
        return REPLIES.index("ask_confirm")
    return REPLIES.index("act_now")


def phi(x: np.ndarray) -> np.ndarray:
    """(n_replies, n_ctx * n_replies) feature matrix for one situation."""
    out = np.zeros((len(REPLIES), len(CTX) * len(REPLIES)))
    for y in range(len(REPLIES)):
        out[y, y * len(CTX) : (y + 1) * len(CTX)] = x
    return out


def logprobs(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = phi(x) @ theta
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def reference_theta() -> np.ndarray:
    """SFT-style start: likes acting right away, a little noise elsewhere."""
    theta = np.zeros(len(CTX) * len(REPLIES))
    theta[REPLIES.index("act_now") * len(CTX) + CTX.index("bias")] = 1.5
    return theta


@dataclass(frozen=True)
class Pair:
    x: np.ndarray
    chosen: int
    rejected: int


def make_pairs(n: int, seed: int, noise: float = 0.10) -> list[Pair]:
    if n < 0 or not 0 <= noise <= 1:
        raise ValueError("n must be >= 0 and noise in [0, 1]")
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        x = situation(rng)
        good = preferred(x)
        bad = int(rng.choice([r for r in range(len(REPLIES)) if r != good]))
        if rng.random() < noise:
            good, bad = bad, good
        pairs.append(Pair(x, good, bad))
    return pairs


def dpo_train(pairs: list[Pair], beta: float = 1.0, lr: float = 2.0, epochs: int = 300) -> tuple[np.ndarray, list[float]]:
    if beta <= 0 or lr <= 0 or epochs < 0:
        raise ValueError("beta and lr must be positive, epochs non-negative")
    ref = reference_theta()
    theta = ref.copy()
    losses = []
    for _ in range(epochs):
        grad = np.zeros_like(theta)
        loss = 0.0
        for p in pairs:
            f = phi(p.x)
            lp, lr_ = logprobs(theta, p.x), logprobs(ref, p.x)
            margin = beta * ((lp[p.chosen] - lr_[p.chosen]) - (lp[p.rejected] - lr_[p.rejected]))
            s = 1.0 / (1.0 + np.exp(-margin))
            loss += -np.log(max(s, 1e-12))
            # d log pi(y) / d theta = phi(y) - E_pi[phi]; the expectation cancels in the difference
            grad += -(1 - s) * beta * (f[p.chosen] - f[p.rejected])
        theta -= lr * grad / len(pairs)
        losses.append(round(float(loss) / len(pairs), 6))
    return theta, losses


def pair_accuracy(theta: np.ndarray, pairs: list[Pair]) -> float:
    """Share of held-out pairs where the policy gives the chosen reply more probability."""
    hits = [logprobs(theta, p.x)[p.chosen] > logprobs(theta, p.x)[p.rejected] for p in pairs]
    return float(sum(hits) / len(hits))


def rule_agreement(theta: np.ndarray, n: int = 2000, seed: int = 99) -> float:
    """Share of fresh situations where the policy's top reply is the house-rule reply."""
    rng = np.random.default_rng(seed)
    xs = [situation(rng) for _ in range(n)]
    return sum(int(np.argmax(logprobs(theta, x))) == preferred(x) for x in xs) / n


def mean_kl(theta: np.ndarray, n: int = 2000, seed: int = 99) -> float:
    rng = np.random.default_rng(seed)
    ref = reference_theta()
    total = 0.0
    for _ in range(n):
        x = situation(rng)
        lp, lr_ = logprobs(theta, x), logprobs(ref, x)
        total += float(np.sum(np.exp(lp) * (lp - lr_)))
    return total / n


def run(seed: int = 3) -> dict[str, float | int]:
    train, test = make_pairs(600, seed), make_pairs(400, seed + 1, noise=0.0)
    ref = reference_theta()
    theta, losses = dpo_train(train)
    return {
        "train_pairs": len(train),
        "test_pairs": len(test),
        "pair_acc_before": round(pair_accuracy(ref, test), 3),
        "pair_acc_after": round(pair_accuracy(theta, test), 3),
        "rule_agree_before": round(rule_agreement(ref), 3),
        "rule_agree_after": round(rule_agreement(theta), 3),
        "kl_to_ref": round(mean_kl(theta), 3),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
    }
