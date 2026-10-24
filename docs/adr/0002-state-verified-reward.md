# ADR 0002: Reward from environment state, not from the agent's words

* **Status:** accepted
* **Context:** the first reward read the final message; the policy learned to say "resolved".
* **Decision:** success, partial credit and penalties are computed from the episode state
  (lookup done, approval given, refund recorded, refund safe). Claims without state are penalised.
  A KL term and a divergence alarm are added.
* **Consequences:** a sparse safe reward made the policy give up, so verified partial credit was
  added. Both runs are kept so the contrast stays visible.
