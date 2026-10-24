# Guardrails and security controls

## PII

* `pii.scrub` masks payment cards (only when the Luhn checksum passes, so order ids and timestamps
  survive), SSNs (only valid-looking ones: no 000, 666 or 9xx area, no 00 group, no 0000 serial),
  email addresses and US-style phone numbers.
* Every split is scrubbed before the teacher labels it, before training, and before anything is
  written to `results/`.
* `ScrubFilter` rewrites log records after argument formatting, so a card passed as a `%s`
  argument is masked too. `get_logger` attaches it once.
* `SafeError` scrubs its message, so validation errors never echo raw values.
* A repo test scrubs every committed `.md`, `.json` and `.csv` file and fails if anything changes.

## Budgets

`Budget` rejects negative, NaN, infinite, boolean and non-numeric values for steps, CPU seconds,
hourly CPU cost and maximum cost, and `cost` / `allows` reject negative runtime inputs. Defaults
are zero cost, since everything here runs locally.

## Training-time safety

* **Reward verified against state.** Success is read from the environment, not from what the
  agent says.
* **Unsafe actions cost more than doing nothing.** A large refund without approval scores below
  giving up.
* **KL penalty** keeps the policy near the reference.
* **Divergence alarm** fires when training reward and verified success separate.
* **DPO beta** bounds how far preference tuning moves the policy; KL is reported.

## Supply chain and repo

Pinned dependency versions, SHA-pinned actions, Dependabot, CodeQL, gitleaks over full history,
secret scanning with push protection, and a brand check whose blocklist is stored only as hashes.
