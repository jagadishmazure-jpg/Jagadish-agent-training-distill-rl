# Results

All data is synthetic (see [how it works](how-it-works.md)). Sample sizes are in each table. These
are regression checks on a toy problem, not claims about real models.

## Distillation

<!-- BEGIN:distill -->
| model | gold test accuracy (n=1,000) | off-template accuracy (n=20) | size |
| --- | --- | --- | --- |
| rules only (57 regex rules) | 95.8% | - | - |
| teacher (rules + 2^18-feature char model) | 99.9% | 45.0% | 14.68 MB |
| student (2^12-feature linear, teacher labels only) | 99.9% | 50.0% | 231 kB |

Student agrees with the teacher on 100.0% of the test set and is 64x smaller.
<!-- END:distill -->

Reading: the student keeps all of the teacher's accuracy on in-distribution data. The 20-message
off-template set shows that neither model generalises well beyond the templates; with real logs
that set would be the first thing to grow.

## DPO

<!-- BEGIN:dpo -->
| metric | reference (start) | after DPO |
| --- | --- | --- |
| held-out pair accuracy (n=400) | 57.0% | 100.0% |
| top reply matches hidden house rule (n=2,000) | 60.4% | 100.0% |
| DPO loss | 0.693 | 0.343 |
| mean KL to reference | 0.000 | 0.427 |
<!-- END:dpo -->

Reading: with 10% noisy pairs, DPO still recovers the house rule. The KL value shows how far the
policy moved from the reference; a smaller beta lets it move further for the same data.

## GRPO-style RL

<!-- BEGIN:grpo -->
| policy | own training reward, first 10 iters | own training reward, last 10 iters | real success | hack rate | unsafe refunds | greedy plan (large refund) |
| --- | --- | --- | --- | --- | --- | --- |
| untrained | - | - | 1.3% | 43.9% | 28.8% | random |
| naive reward | 0.544 | 0.948 | 0.0% | 99.3% | 0.4% | claim_resolved |
| guarded reward | -0.704 | 1.397 | 99.1% | 0.2% | 0.2% | lookup_order -> ask_approval -> issue_refund -> reply_done |

Hacking alarm (proxy minus real success > 0.5 over 10 iterations): naive run trips at iteration 9; guarded run: never.
<!-- END:grpo -->

Reading: the naive run's own reward goes up while real success stays at zero, which is reward
hacking. Each run's training reward is on its own scale, so compare real success and hack rate
across runs, not the reward columns. Curve: [reward_curve.svg](../results/reward_curve.svg).

## Serving latency (recorded)

<!-- BEGIN:bench -->
| model | accuracy (n=1,000) | serialized size | p50 latency | p95 latency |
| --- | --- | --- | --- | --- |
| teacher | see above | see above | 2648 us | 3162 us |
| student | see above | see above | 222 us | 282 us |
| MLP fp32 | 99.9% | 9.46 MB | 324 us | 388 us |
| MLP int8 (dynamic) | 99.9% | 2.37 MB | 153 us | 182 us |

Recorded on the build box: Python 3.13.5, torch 2.5.1+cpu, x86_64, 1 thread, batch size 1, 900 timed calls per row. Largest fp32-vs-int8 logit gap: 0.4583.
<!-- END:bench -->

Latency depends on the machine. Re-run `distillrl bench` to measure on yours.

Test count at last render: <!-- BEGIN:tests -->128<!-- END:tests -->.
