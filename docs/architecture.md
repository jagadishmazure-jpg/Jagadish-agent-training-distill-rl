# Architecture

```mermaid
flowchart TB
  subgraph data[Data, src/distillrl/data.py]
    GEN[Seeded generator\n7 tools, typos, distractors, fake PII] --> SCR[pii.scrub]
    SCR --> SEED[500 gold seeds]
    SCR --> POOL[4,000 unlabelled]
    SCR --> TEST[1,000 gold test]
  end
  subgraph distil[Distillation]
    SEED --> TEA[teacher.Teacher\n57 rules + char 1-5 gram LR, 2^18 features]
    POOL --> TEA
    TEA -->|hard labels + confidence weights| STU[student.Student\nchar 2-4 gram LR, 2^12 features]
  end
  subgraph serve[Serving, quant.py]
    TEA -->|labels| MLP[ToolMLP fp32\n4096-512-512-7]
    MLP --> INT8[quantize_dynamic int8]
  end
  subgraph pref[Preference tuning]
    PAIRS[make_pairs: 600 noisy pairs] --> DPO[dpo_train\nfrozen reference]
    ENV[grpo.Episode refund env] --> NAIVE[naive_reward]
    ENV --> GUARD[guarded_reward + KL]
    NAIVE --> RUNA[GRPO run A]
    GUARD --> RUNB[GRPO run B]
    RUNA --> ALARM[hacking_alarm]
    RUNB --> ALARM
  end
  TEST --> REPORT[report.full / bench.run]
  STU --> REPORT
  INT8 --> REPORT
  DPO --> REPORT
  ALARM --> REPORT
  REPORT --> DOCS[render_docs.py -> README]
```

## Components

| Module | Role |
|---|---|
| `data.py` | seeded synthetic messages, gold tools, three disjoint splits, scrubbed |
| `pii.py` | card (Luhn), SSN, email and phone masking; logging filter; `SafeError` |
| `teacher.py` | rule scorer and the big char model, averaged 50/50 |
| `student.py` | distilled linear student, accuracy helper |
| `quant.py` | torch MLP, int8 conversion, size and logit-gap helpers |
| `dpo.py` | toy reply policy, preference pairs, DPO loss and gradient |
| `grpo.py` | refund environment, two rewards, GRPO-style trainer, alarm |
| `budget.py` | run budget that rejects negative or non-finite values |
| `report.py`, `bench.py`, `cli.py`, `plot.py` | runs, recorded benchmark, tables, SVG curve |

## Determinism

All random generators are seeded. Accuracy, size and reward numbers are reproducible for the pinned
dependency set and are checked in CI. Latency is not reproducible, so it is recorded once in
`results/bench.json` with the machine description and rendered from there.
