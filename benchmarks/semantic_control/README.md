# Semantic Control Prototype v1 — Contract Freeze

This directory freezes the interface and benchmark boundary before semantic-engine implementation.

## What is authoritative

The fixtures here are derived copies for testing. Scientific truth remains in research-engine authoritative records. Task/policy/execution authority remains in workspace-control authoritative records. Exact source repository, commit, path, and Git blob identities are pinned in `fixture-manifest.v1.json`.

## Frozen V1 interfaces

- `SemanticObservationV1`: provenance-bearing typed semantic assertion.
- `EvidenceRequirementV1`: request-only evidence need.
- `OntologySliceV1`: deterministic bounded schema/context slice.
- `CandidateActionV1`: proposed action only; authorization is always `UNDECIDED`.
- `DecisionEvaluationV1`: separate policy-bound evaluation with `ALLOW`, `DENY`, or `ABSTAIN`.

## Safety boundary

Semantic state can suggest work, expose contradictions, and request more evidence. It cannot grant execution authority. Any future `ALLOW` must bind explicit authority-owned policy context. Missing required authority context fails closed.

No production EXTRACT/VERIFY, database schema, policy, model, Lambda, or executor mutation is part of this freeze.
