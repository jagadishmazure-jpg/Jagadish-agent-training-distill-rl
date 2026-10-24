# Purpose

## The question this repo answers

Once an agent works with a large model, three follow-up questions arrive quickly:

1. Can a much smaller model do the same routing job, so each request is cheaper and faster?
2. Can we steer behaviour (ask before risky actions, refuse injected instructions) from
   preference data instead of hand-written prompts?
3. If we reward the agent for finishing tasks, will it learn to finish them, or learn to look finished?

This lab answers each one with a small, measured experiment that runs on a CPU for free.

## Who it is for

* **Recruiters and hiring managers:** a compact proof that the author understands distillation,
  DPO, GRPO-style RL, reward hacking and quantisation, with numbers from real runs.
* **Engineers adopting the ideas:** the code is short and dependency-light (numpy, scikit-learn,
  torch CPU). Swap the synthetic data for your own tool-call logs; see the
  [implementation guide](implementation-guide.md).
* **The owner:** a reference for interviews, see [interview points](interview-points.md).

## The scenario

Brindlewood Gear is a fictional outdoor-gear retailer. Its support agent reads a customer message and
chooses one of seven tools: `lookup_order`, `track_shipment`, `issue_refund`, `update_address`,
`cancel_order`, `escalate_human`, `answer_faq`. Refunds above $50 need the customer's approval.

## What it is not

* Not a benchmark of real language models. The data is synthetic and template-generated.
* Not production training code. It shows the mechanics and the failure modes at toy scale.
