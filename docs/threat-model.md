# Threat model

Scope: the training pipeline (data, teacher, student, DPO, RL) and the served small model.

## STRIDE

| Threat | Example here | Control |
|---|---|---|
| Spoofing | forged preference pairs or labels injected into training data | gold set kept separate; pairs evaluated on a clean held-out set |
| Tampering | poisoned logs shift the student | confidence weighting, gold and off-template checks before release |
| Repudiation | cannot tell which data trained a model | seeded, deterministic runs; numbers regenerated in CI |
| Information disclosure | PII in logs memorised by the model or written to logs | scrub before labelling and training; log filter; `SafeError` |
| Denial of service | runaway training cost | `Budget` with non-negative caps; CPU-only |
| Elevation of privilege | RL policy learns to issue refunds without approval | unsafe actions penalised; verified-state reward; tests on the final plan |

## OWASP Top 10 for LLM applications (relevant items)

| Risk | Relevance | Control |
|---|---|---|
| LLM01 Prompt injection | DPO teaches `refuse` when injection is present | house-rule test on fresh situations |
| LLM02 Sensitive information disclosure | training data with cards and emails | scrubber, repo-wide PII test |
| LLM03 Supply chain | model and package downloads | pinned versions, Dependabot, CodeQL |
| LLM04 Data and model poisoning | bad labels or pairs | gold set, noisy-pair robustness, off-template check |
| LLM06 Excessive agency | refunds without approval | approval required in the reward and the evaluation |
| LLM09 Misinformation | agent claims success it did not achieve | `claim_resolved` without state is scored as a hack |
