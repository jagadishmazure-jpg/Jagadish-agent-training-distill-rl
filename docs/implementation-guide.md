# Implementation guide (for adopters)

## Distil your own router

1. Export tool-call logs: the user message and the tool your big model chose. Scrub them with
   `pii.scrub` (or a stronger detector) before they leave the logging system.
2. Keep a gold set labelled by people, separate from the logs. At least a few hundred per tool.
3. Use your current big model as the teacher. Store its confidence with each label.
4. Train the student with confidence weights. Start linear; move to a small transformer only if
   the gold-set gap is too large.
5. Accept the student only if gold accuracy is within your tolerance of the teacher **and** an
   off-distribution set (new phrasings, new products) does not regress.
6. Route low-confidence student predictions back to the teacher (a cascade).

## Preference tuning

1. Turn human reviews or approval clicks into (chosen, rejected) pairs.
2. Freeze a reference model; tune with DPO; report KL to the reference along with win rate.
3. Use held-out pairs with no noise for evaluation, and a fixed rule-based check for safety
   behaviours (asks before refunds, refuses injected instructions).

## RL with trajectory rewards

1. Define success from **state** (database rows, API side effects), never from the agent's text.
2. Penalise unsafe actions more than inaction, and claims not backed by state.
3. Add partial credit only for verified steps, otherwise the safe optimum can be "do nothing".
4. Track training reward and verified success side by side; alert when they diverge.

## Quantise for serving

1. Score fp32 and int8 on the same gold set; block release if accuracy drops more than your budget.
2. Time p50 and p95 at batch size 1 on the target hardware; record versions and thread count.

## Mapping to this code

| Step | Function |
|---|---|
| scrub | `pii.scrub`, `data.scrubbed` |
| teacher labels | `teacher.Teacher.label_many`, `student.teacher_labels` |
| student | `student.train` |
| DPO | `dpo.dpo_train`, `dpo.pair_accuracy`, `dpo.mean_kl` |
| RL | `grpo.train`, `grpo.guarded_reward`, `grpo.hacking_alarm` |
| int8 | `quant.to_int8`, `bench.run` |
