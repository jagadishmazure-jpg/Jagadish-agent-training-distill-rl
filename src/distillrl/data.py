"""Synthetic tool-call requests for a fictional outdoor-gear retailer, Brindlewood Gear.

Each example is a customer message plus the gold tool the support agent should call.
Messages are built from templates with paraphrases, filler, typos and a second "distractor"
topic, so a keyword rule set gets most but not all of them right. Some messages carry fake
PII (Luhn-valid test card numbers, emails, phones) so the scrubber is exercised.
Everything is seeded and fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .pii import scrub

TOOLS = (
    "lookup_order",
    "track_shipment",
    "issue_refund",
    "update_address",
    "cancel_order",
    "escalate_human",
    "answer_faq",
)

TEMPLATES: dict[str, list[str]] = {
    "lookup_order": [
        "can you pull up order {oid}",
        "what is in my order {oid}",
        "show me the details for {oid}",
        "i need the receipt for order {oid}",
        "did order {oid} go through",
        "check what i bought on {oid}",
        "what did i pay for {oid}",
    ],
    "track_shipment": [
        "where is my package {oid}",
        "has {oid} shipped yet",
        "tracking for {oid} please",
        "when will {oid} arrive",
        "my parcel {oid} is late",
        "is the courier bringing {oid} today",
        "the box for {oid} has not shown up",
    ],
    "issue_refund": [
        "i want my money back for {oid}",
        "please refund {oid}",
        "the {item} was broken, refund me",
        "charge on my card for {oid} should be reversed",
        "return the {item} and give me a refund",
        "i was charged twice for {oid}",
        "credit my account for the faulty {item}",
    ],
    "update_address": [
        "change the delivery address on {oid}",
        "i moved, update my shipping address",
        "send {oid} to my new place instead",
        "wrong street on {oid}, please fix it",
        "update address for my account",
        "deliver {oid} to my office not home",
    ],
    "cancel_order": [
        "cancel {oid}",
        "i do not want {oid} anymore, stop it",
        "please cancel my order before it ships",
        "call off order {oid}",
        "kill the order {oid} i placed by mistake",
        "i ordered the wrong {item}, scrap it",
    ],
    "escalate_human": [
        "let me talk to a real person",
        "this is the third time, get me a manager",
        "i want to speak to someone now",
        "your bot is useless, human please",
        "i am going to dispute this with my bank unless someone calls me",
        "connect me with an agent",
    ],
    "answer_faq": [
        "what is your return policy",
        "do you ship to canada",
        "how long does a {item} warranty last",
        "are your stores open on sunday",
        "what sizes does the {item} come in",
        "how do i wash a {item}",
        "do you price match",
    ],
}

ITEMS = ["tent", "sleeping bag", "rain jacket", "trail shoe", "headlamp", "backpack", "camp stove"]
FILLER = ["hi", "hello", "hey there", "um", "quick question", "sorry to bother", "", "", ""]
TAILS = ["thanks", "asap", "thank you", "pls", "", "", "", "my email is {email}", "call me at {phone}"]
DISTRACT = [
    "also love the {item}",
    "btw your site is slow",
    "the {item} i got last year was great",
    "not about shipping really",
    "no refund needed",
]
TEST_CARDS = ["4111 1111 1111 1111", "5555 5555 5555 4444", "3782 822463 10005"]


@dataclass(frozen=True)
class Example:
    text: str
    tool: str


def _typo(rng: random.Random, text: str) -> str:
    if len(text) < 6 or rng.random() > 0.25:
        return text
    i = rng.randrange(1, len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def make_example(rng: random.Random) -> Example:
    tool = rng.choice(TOOLS)
    body = rng.choice(TEMPLATES[tool]).format(oid=f"BW-{rng.randint(10000, 99999)}", item=rng.choice(ITEMS))
    parts = [rng.choice(FILLER), _typo(rng, body)]
    if rng.random() < 0.3:
        parts.append(rng.choice(DISTRACT).format(item=rng.choice(ITEMS)))
    tail = rng.choice(TAILS).format(email=f"cust{rng.randint(1, 999)}@example.com", phone="555-201-4477")
    parts.append(tail)
    if tool == "issue_refund" and rng.random() < 0.15:
        parts.append(f"card {rng.choice(TEST_CARDS)}")
    text = " ".join(p for p in parts if p).strip()
    return Example(text=text, tool=tool)


def make_dataset(n: int, seed: int) -> list[Example]:
    if n < 0:
        raise ValueError("n must not be negative")
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(n)]


def scrubbed(rows: list[Example]) -> list[Example]:
    """PII is removed before any row is labelled, trained on, or written to disk."""
    return [Example(text=scrub(r.text), tool=r.tool) for r in rows]


SEED_SIZE, POOL_SIZE, TEST_SIZE = 500, 4000, 1000


def splits(seed: int = 7) -> tuple[list[Example], list[Example], list[Example]]:
    """(gold seed set for the teacher, unlabelled pool, gold test set). All scrubbed, all disjoint draws."""
    return (
        scrubbed(make_dataset(SEED_SIZE, seed + 2000)),
        scrubbed(make_dataset(POOL_SIZE, seed)),
        scrubbed(make_dataset(TEST_SIZE, seed + 1000)),
    )
