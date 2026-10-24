# ADR 0003: Record latency instead of checking it in CI

* **Status:** accepted
* **Context:** latency differs between machines and runs; a CI equality check would always fail.
* **Decision:** deterministic numbers (accuracy, sizes, rewards, test count) are re-run and compared
  in CI. Latency is measured by `distillrl bench`, saved with the machine description in
  `results/bench.json`, and rendered from that file. Tests check only stable relations
  (int8 smaller than fp32, student faster than teacher, accuracy within 2 points).
* **Consequences:** latency in the README is honest for one machine and labelled as such.
