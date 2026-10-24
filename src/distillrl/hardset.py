"""Twenty hand-written messages that do not follow any generator template.

They show how far the distilled student generalises beyond the synthetic distribution.
This is a tiny regression check, not a quality claim; misses stay visible in the report.
"""

from __future__ import annotations

from .data import Example

HARD = [
    Example("the stove i got never turned on, i expect to be reimbursed", "issue_refund"),
    Example("any idea which truck has my stuff and when it gets here", "track_shipment"),
    Example("please put a stop on BW-55120 i found it cheaper elsewhere", "cancel_order"),
    Example("i relocated to another city last week", "update_address"),
    Example("is there someone with a pulse i can chat with", "escalate_human"),
    Example("can i bring a jacket back if it does not fit", "answer_faq"),
    Example("what items were on BW-33017", "lookup_order"),
    Example("BW-77002 says delivered but nothing at my door", "track_shipment"),
    Example("i need the cash returned to my card for the leaking tent", "issue_refund"),
    Example("my zip code is wrong on the last purchase", "update_address"),
    Example("forget the backpack order, i changed my mind", "cancel_order"),
    Example("supervisor please", "escalate_human"),
    Example("are gift cards accepted online", "answer_faq"),
    Example("how much did the headlamp on BW-10442 cost me", "lookup_order"),
    Example("is my delivery stuck somewhere", "track_shipment"),
    Example("the zipper snapped on day one, want reimbursement", "issue_refund"),
    Example("ship it to my parents house please, BW-61200", "update_address"),
    Example("i no longer need BW-80301", "cancel_order"),
    Example("i have asked four times, escalate this", "escalate_human"),
    Example("do your sleeping bags come in kids sizes", "answer_faq"),
]
