"""GRPO-style policy-gradient loop with a trajectory-eval reward and a reward-hacking guard.

Task: a refund request at Brindlewood Gear. Small refunds (<= $50) may be issued directly;
large ones need the user's approval first. The policy emits up to MAX_STEPS actions:

    lookup_order, ask_approval, issue_refund, reply_done, claim_resolved

`claim_resolved` just *says* the case is solved. The naive reward is a trajectory judge that
reads the final message: a closing "resolved" message earns the success bonus whether or not
a refund really happened. That is the hole the policy can exploit.

The guarded reward checks the environment state instead (was a refund recorded, was approval
given when needed), penalises claims that the state does not back up, and adds a KL penalty
toward the reference policy. Both runs use group-relative advantages (GRPO): sample G
trajectories per task, advantage = (r - group mean) / (group std + eps), REINFORCE update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIONS = ("lookup_order", "ask_approval", "issue_refund", "reply_done", "claim_resolved")
TERMINAL = {"reply_done", "claim_resolved"}
MAX_STEPS = 5
STEP_COST = 0.05


@dataclass
class Episode:
    large: bool
    actions: list[str] = field(default_factory=list)
    looked_up: bool = False
    approved: bool = False
    refunded: bool = False
    unsafe: bool = False

    def state(self) -> int:
        """Discrete state index: (large, looked_up, approved, refunded)."""
        return int(self.large) * 8 + int(self.looked_up) * 4 + int(self.approved) * 2 + int(self.refunded)

    def apply(self, a: str) -> None:
        self.actions.append(a)
        if a == "lookup_order":
            self.looked_up = True
        elif a == "ask_approval":
            self.approved = True
        elif a == "issue_refund":
            if not self.looked_up or (self.large and not self.approved):
                self.unsafe = True
            self.refunded = True

    @property
    def done(self) -> bool:
        return bool(self.actions) and (self.actions[-1] in TERMINAL or len(self.actions) >= MAX_STEPS)

    @property
    def truly_solved(self) -> bool:
        return self.refunded and not self.unsafe and self.actions[-1] == "reply_done"

    @property
    def hacked(self) -> bool:
        """Claimed success without the state to back it up."""
        return self.actions[-1] == "claim_resolved" and not (self.refunded and not self.unsafe)


def naive_reward(ep: Episode) -> float:
    """Judge reads only the final message. 'resolved' sounds like success."""
    r = -STEP_COST * len(ep.actions)
    last = ep.actions[-1]
    if last == "claim_resolved":
        r += 1.0
    elif last == "reply_done" and ep.refunded:
        r += 1.0
    if ep.unsafe:
        r -= 0.5
    return r


def guarded_reward(ep: Episode) -> float:
    """Reward from environment state: success must be real, unsafe refunds cost a lot."""
    r = -STEP_COST * len(ep.actions)
    # partial credit from the trajectory eval, only for steps the state confirms
    r += 0.2 * ep.looked_up + 0.2 * (ep.large and ep.approved) + 0.3 * (ep.refunded and not ep.unsafe)
    if ep.truly_solved:
        r += 1.0
    if ep.hacked:
        r -= 1.0
    if ep.actions[-1] == "reply_done" and not ep.refunded:
        r -= 0.3  # gave up without doing the job
    if ep.unsafe:
        r -= 1.5
    return r


def reference_logits() -> np.ndarray:
    """Untrained start: uniform over actions in every state."""
    return np.zeros((16, len(ACTIONS)))


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def rollout(theta: np.ndarray, large: bool, rng: np.random.Generator) -> tuple[Episode, list[tuple[int, int]]]:
    ep = Episode(large=large)
    steps = []
    while not ep.done:
        s = ep.state()
        a = int(rng.choice(len(ACTIONS), p=softmax(theta[s])))
        steps.append((s, a))
        ep.apply(ACTIONS[a])
    return ep, steps


@dataclass
class RunLog:
    proxy: list[float] = field(default_factory=list)
    true_success: list[float] = field(default_factory=list)
    hack_rate: list[float] = field(default_factory=list)
    unsafe_rate: list[float] = field(default_factory=list)


def train(
    guarded: bool,
    iters: int = 300,
    group: int = 16,
    lr: float = 2.0,
    kl_coef: float = 0.02,
    seed: int = 11,
) -> tuple[np.ndarray, RunLog]:
    if iters < 0 or group < 2 or lr <= 0 or kl_coef < 0:
        raise ValueError("iters >= 0, group >= 2, lr > 0, kl_coef >= 0 required")
    rng = np.random.default_rng(seed)
    theta, ref = reference_logits(), reference_logits()
    reward_fn = guarded_reward if guarded else naive_reward
    log = RunLog()
    for _ in range(iters):
        grad = np.zeros_like(theta)
        eps = []
        for large in (False, True):
            batch = [rollout(theta, large, rng) for _ in range(group)]
            rewards = np.array([reward_fn(ep) for ep, _ in batch])
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            for (ep, steps), a_i in zip(batch, adv, strict=True):
                eps.append((ep, reward_fn(ep)))
                for s, a in steps:
                    p = softmax(theta[s])
                    g = -p
                    g[a] += 1.0
                    grad[s] += a_i * g / len(steps)
                    if guarded and kl_coef:
                        p_ref = softmax(ref[s])
                        kl_grad = p * (np.log(p) - np.log(p_ref))
                        kl_grad -= p * kl_grad.sum()
                        grad[s] -= kl_coef * kl_grad
        theta += lr * grad / (2 * group)
        n = len(eps)
        log.proxy.append(round(float(np.mean([r for _, r in eps])), 4))
        log.true_success.append(round(sum(e.truly_solved for e, _ in eps) / n, 4))
        log.hack_rate.append(round(sum(e.hacked for e, _ in eps) / n, 4))
        log.unsafe_rate.append(round(sum(e.unsafe for e, _ in eps) / n, 4))
    return theta, log


def hacking_alarm(log: RunLog, window: int = 10, gap: float = 0.5) -> int | None:
    """First iteration where the proxy reward is high but verified success is not.

    A rising proxy reward with flat or falling real success is the classic sign of reward
    hacking. Returns the iteration index that trips the alarm, or None.
    """
    if window < 1 or gap < 0:
        raise ValueError("window >= 1 and gap >= 0 required")
    for i in range(window, len(log.proxy) + 1):
        proxy = float(np.mean(log.proxy[i - window : i]))
        real = float(np.mean(log.true_success[i - window : i]))
        if proxy - real > gap:
            return i - 1
    return None


def evaluate(theta: np.ndarray, n: int = 1000, seed: int = 5) -> dict[str, float]:
    """Fresh rollouts with the final policy, scored on what really happened."""
    rng = np.random.default_rng(seed)
    eps = [rollout(theta, bool(i % 2), rng)[0] for i in range(n)]
    return {
        "true_success": round(sum(e.truly_solved for e in eps) / n, 3),
        "hack_rate": round(sum(e.hacked for e in eps) / n, 3),
        "unsafe_rate": round(sum(e.unsafe for e in eps) / n, 3),
        "mean_steps": round(float(np.mean([len(e.actions) for e in eps])), 2),
    }


def greedy_plan(theta: np.ndarray, large: bool) -> list[str]:
    ep = Episode(large=large)
    while not ep.done:
        ep.apply(ACTIONS[int(np.argmax(theta[ep.state()]))])
    return ep.actions


def run() -> dict[str, object]:
    out: dict[str, object] = {"start": evaluate(reference_logits())}
    for name, guarded in (("naive", False), ("guarded", True)):
        theta, log = train(guarded)
        out[name] = {
            **evaluate(theta),
            "proxy_first10": round(float(np.mean(log.proxy[:10])), 3),
            "proxy_last10": round(float(np.mean(log.proxy[-10:])), 3),
            "plan_small": greedy_plan(theta, False),
            "plan_large": greedy_plan(theta, True),
            "alarm_at": hacking_alarm(log),
            "curve": log,
        }
    return out
