# ADR 0001: CPU-only, toy-scale experiments

* **Status:** accepted
* **Context:** the portfolio must stay free (no paid APIs, no GPUs) and CI must finish in minutes.
* **Decision:** use numpy, scikit-learn and CPU torch; keep models in the kilobyte-to-megabyte range;
  make every experiment deterministic and seeded.
* **Consequences:** the numbers show mechanics, not real language-model quality. Bigger runs are
  documented as optional and never run in CI.
