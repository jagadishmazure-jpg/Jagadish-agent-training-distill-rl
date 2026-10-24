# Interview points

Repo size at last render: <!-- BEGIN:tests -->128<!-- END:tests --> tests.

## Thirty-second version

"I distilled a 14.7 MB tool-routing teacher into a 231 kB student with no accuracy loss on 1,000
gold messages, tuned reply behaviour with DPO, and showed reward hacking live: a naive judge reward
drove real success to zero while the policy claimed success 99% of the time; a state-verified reward
got 99% real success. Then int8 quantisation cut the serving model 4x with no accuracy loss. All on
CPU, all numbers from real runs."

## Points to make

* **Distillation is about cost and latency, not accuracy.** The student is 64x smaller and about
  12x faster per request; it inherits the teacher's errors.
* **Confidence weighting** reduces the pull of uncertain teacher labels.
* **The honest weak spot:** both models score about 50% off-template. Template data overstates quality.
* **DPO vs RLHF with PPO:** DPO needs no separate reward model or sampling loop; the reference
  model and beta stand in for the KL constraint.
* **GRPO:** advantage is the reward standardised within a group of samples for the same prompt, so
  there is no value network.
* **Reward hacking:** the policy optimises the reward you wrote, not the one you meant. Verify
  against state, penalise unsupported claims, add KL, and watch reward vs real success.
* **Sparse safe rewards can make the policy give up.** Partial credit for verified steps fixed it.
* **Dynamic int8** quantises weights ahead of time and activations at run time; it suits
  Linear-heavy models on CPU.

## Likely questions

* *How would you pick the student size?* Sweep sizes, plot gold accuracy against latency, and pick
  the smallest within tolerance, then check the off-distribution set.
* *What if the teacher is wrong?* Keep a human gold set; use confidence to route back to the
  teacher; consider soft labels.
* *How do you detect reward hacking in production?* Sample trajectories for human review, compare
  judge scores with downstream outcomes, alert on divergence.
* *Why not vLLM here?* It pays off on GPUs; this lab stays free. See serving-runners.md.
