# How it works, step by step

## 1. Data

1. `make_example` picks a tool, fills one of its templates with an order id (`BW-12345`) or item,
   adds optional filler, a typo (25% chance of swapping two letters), a distractor clause
   (30%) and an optional tail that may contain a fake email or phone number. Some refund messages
   carry a public test card number.
2. `splits()` draws three independent sets and runs every row through `pii.scrub`, so the teacher,
   student and test set only ever see `[CARD]`, `[EMAIL]` and `[PHONE]` placeholders.

## 2. Distillation

1. **Teacher.** Two scorers are averaged: 57 weighted regex rules turned into a probability
   distribution with a softmax, and a logistic model on character 1-5 grams hashed into 2^18
   features, fitted on the 500 gold seed messages. Rules alone reach 95.8% (typos break them);
   the combination reaches 99.9%.
2. **Labelling.** The teacher labels the 4,000-message pool. Gold labels in the pool are ignored
   (a test proves this by feeding a deliberately wrong gold label).
3. **Student.** A logistic model on character 2-4 grams hashed into only 2^12 features is trained on
   the teacher's labels, each weighted by the teacher's confidence (floored at 0.2) so uncertain
   labels count less.
4. **Measurement.** Accuracy against the 1,000 gold test messages, agreement with the teacher,
   pickled weight size, and single-request latency (median and p95 over 900 timed calls, 1 thread).
5. **Generalisation check.** 20 hand-written messages that follow no template. Both models score
   around 50%. This is the honest limit of template data and stays visible as a test.

## 3. DPO

1. Each situation has four features (risky action, user consented, injection present, user angry)
   plus a bias. A hidden house rule picks the preferred reply among `act_now`, `ask_confirm`,
   `refuse`, `hand_off`.
2. 600 training pairs (chosen, rejected) are drawn from that rule with 10% of pairs flipped as noise.
3. The policy is a softmax over a linear score. The reference policy starts by preferring `act_now`.
4. `dpo_train` minimises `-log sigmoid(beta * (chosen log-ratio - rejected log-ratio))` with plain
   gradient descent. The analytic gradient is checked against finite differences in the tests.
5. Results: held-out pair accuracy, how often the top reply equals the house rule on 2,000 fresh
   situations, and mean KL divergence from the reference (how far the policy moved).

## 4. GRPO-style RL and reward hacking

1. **Environment.** A refund episode, small (approval not needed) or large (approval needed).
   Actions: `lookup_order`, `ask_approval`, `issue_refund`, `reply_done`, `claim_resolved`, at most 5.
   An unsafe refund is one issued before lookup, or a large one without approval.
2. **Naive reward.** A judge that reads the final message: `claim_resolved` earns the success bonus
   regardless of state. Steps cost 0.05 each.
3. **Guarded reward.** Success only when the state shows a safe refund followed by `reply_done`;
   partial credit only for steps the state confirms; -1 for claims the state does not back up,
   -0.3 for giving up, -1.5 for unsafe refunds; plus a KL penalty (0.02) toward the reference.
4. **Training.** Each iteration samples a group of 16 trajectories per task type, standardises
   rewards within the group (the GRPO advantage) and takes a REINFORCE step. 300 iterations.
5. **Alarm.** `hacking_alarm` fires at the first 10-iteration window where mean training reward
   exceeds mean verified success by more than 0.5.
6. **Evaluation.** 1,000 fresh rollouts per policy scored on what really happened.

## 5. int8 vs fp32 serving

1. A 4096-512-512-7 MLP is trained on teacher labels (8 epochs, Adam).
2. `torch.ao.quantization.quantize_dynamic` converts the Linear layers to int8 weights.
3. Both are scored on the gold test set, serialised with `torch.save` for size, and timed one
   request at a time on a single thread.
