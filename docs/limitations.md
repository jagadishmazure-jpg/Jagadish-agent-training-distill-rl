# Limitations and known failures

* **Template data.** In-distribution accuracy is near perfect because the test set comes from the
  same templates. On 20 hand-written off-template messages both teacher and student score about
  50%. A test asserts this stays low, so the weakness cannot be hidden by accident.
  Typical misses: "reimbursed" (no refund keyword), "put a stop on" (cancel), "relocated" (address).
* **The teacher is not a language model.** It stands in for one: bigger, slower, more accurate than
  the rules alone. Real distillation from a hosted model would also need to handle its mistakes
  and cost.
* **Student beats nothing it was not taught.** Agreement with the teacher is 100% on the test set,
  so the student inherits teacher errors too.
* **DPO is on a toy linear policy.** It shows the loss, the reference model and beta, not language
  quality. A real run would use a small open model.
* **The GRPO environment is tiny** (16 states, 5 actions). Reward shaping (partial credit) was
  needed: without it, the guarded run settled on giving up (about 1% real success in an earlier
  version). This is a real lesson: a safe reward that is too sparse can make the policy do nothing.
* **Latency is machine-dependent** and recorded once; dynamic int8 helps Linear-heavy models on CPU
  but gains vary with hardware and batch size.
* **No real model serving.** vLLM and llama.cpp are documented, not run.
