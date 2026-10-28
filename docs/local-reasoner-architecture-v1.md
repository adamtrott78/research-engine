# Local Reasoner Architecture v1

This design selects the local reasoning tier **from the research-engine workload and Lambda constraints**, rather than choosing a model from public leaderboards.

## Decision

The first production-shaped experiment uses **one physical local reasoner owner, concurrency 1**, with fresh logical contexts per invocation. The first-wave bakeoff is:

1. existing certified Qwen3.5-4B baseline;
2. official Qwen3.5-9B as the parameter-count/control arm;
3. Qwythos-9B-v2 as the reasoning-distillation arm.

Qwen3.6-35B-A3B is second-wave only if the 9B class fails the frozen workload gates.

No candidate is promoted by this design.

## Why one owner

The current controller admits one active research run by default, and the orchestration graph is already bounded. With N retained papers, the current graph has at most approximately `10 + 2*N` semantic jobs under default fanout, but only a subset should use deep local reasoning.

The reasoner is reserved for:

- planning and critique;
- unresolved local hard cases;
- synthesis and gap generation;
- novelty arbitration;
- explicit critic/reviewer calls.

Deterministic acquisition/checkpointing, document parsing, formula/table inference, baseline retrieval, and the already accepted bounded Qwen3.5-4B VERIFY path stay separate.

Multiple logical agents do **not** require multiple model copies. Each call can receive a fresh prompt/context while sharing one immutable resident model owner.

## Context policy

Normal reasoning: 16K context / <=4K output.

Deep reasoning: 32K context / <=8K output.

The model's giant advertised context is not the research database. PostgreSQL, checkpoints, source artifacts and retrieval remain durable memory. Long-context inference is used only when a bounded evidence packet justifies it.

## GPU policy

The first bakeoff does not attempt co-residency. GitRunner exposes one selected A4500 to the candidate, while the other device is denied. This produces honest model-specific resource numbers and avoids disturbing the existing formula/table/Qwen VERIFY ownership policy.

A first-wave candidate must remain below 16 GiB peak device memory, with <=14 GiB preferred. Co-residency or production family switching is a later measured integration question, not an assumption.

The default target is therefore a single-GPU quantized 9B-class text reasoner. A 35B-A3B model is treated as an occasional exclusive deep-mode candidate, not a permanent resident.

## What decides the winner

Generic benchmark scores are background evidence only. The winner is determined on our failure modes:

- obligation validity / atomic fact support;
- exact provenance binding;
- bounded source-grounded verification;
- semantic-control fixtures;
- synthesis/gap/novelty reasoning;
- wrong-hypothesis recovery;
- adversarial critique/blocker detection.

False acceptance, fabricated provenance, wrong source identity, cloud fallback, scientific-state mutation, or resource-ceiling violations are hard failures.

The economic metric is **successful research work per GPU-second**, not tokens/second alone.

## 40-minute autonomous test

Only finalists receive the endurance test:

- hard wall clock: 2400 seconds;
- one physical model owner;
- concurrency 1;
- zero OpenAI/ChatGPT/Copilot model calls;
- zero cloud-model fallback;
- frozen local evidence/tool fixtures;
- durable checkpoints;
- no scientific-state materialization.

Telemetry separates actual model-active seconds from tool/database/wait time. This tells us whether a 40-minute research loop truly needs 40 minutes of GPU decoding or only intermittent reasoning bursts.

## Iteration policy

This is **experimental research architecture**. The same executor may:

design -> implement -> run -> inspect failure -> change the hypothesis -> rerun.

There is no fresh independent-review requirement for each iteration. Correctness comes from frozen pre-result contracts, exact Git heads, CI, immutable model/runtime identities and machine-readable experiment receipts.

A failed experiment should make the next iteration faster, not trigger minutes of reviewer-chat ceremony.

Production promotion is a separate boundary. If a result would change scientific-state authority, production routing, the accepted VERIFY path, long-lived model services, or live Lambda/control-plane behavior, stop and obtain explicit user authorization. The governing task can then choose whether independent review is worth its latency for that blast radius.

## GitRunner sequence

1. CPU design validation.
2. Immutable model/runtime admission.
3. One-candidate-at-a-time single-GPU resource probe.
4. Frozen quality bakeoff.
5. 40-minute local-only endurance run for finalists.
6. Machine-readable decision.

The architecture contract in `benchmarks/semantic_control/local-reasoner-architecture.v1.json` is the machine-readable authority for this experiment.
