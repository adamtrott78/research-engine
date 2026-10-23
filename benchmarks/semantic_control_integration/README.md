# Semantic Control Integration v1 — Contract and Baseline Freeze

This directory freezes the **real integration boundary** before any integration adapter is implemented.

## What is being measured

The relational baseline uses the existing public `PostgresStore` read methods against the repository's existing five-work corpus in the ephemeral PostgreSQL service already provided by normal CI.

The measured call includes current connection/open-close behavior because that is part of the existing public store path. Database initialization and fixture loading are setup only and are excluded from timings.

There is deliberately **no latency pass/fail threshold** at this stage. The baseline records empirical latency and answer correctness. Semantic overhead will be compared later without inventing a target after seeing the numbers.

## What stays authoritative

- research-engine PostgreSQL/CAS and authoritative records remain scientific truth.
- workspace-control task/policy/project/execution artifacts remain task and authority truth.
- semantic projections remain deterministic, rebuildable, read-only derived views.
- CandidateAction remains non-authoritative; policy-bound DecisionEvaluation remains the only authorization bridge.

## EXTRACT dependency

Local EXTRACT is still an independent scientific task. Current Wave7/Family-C artifacts do **not** define this integration contract.

The EXTRACT-specific integration stage remains gated until the owning task durably records an **accepted** architecture/output checkpoint with exact artifact/schema identity. The integration layer will adapt to that accepted output; EXTRACT will not be redesigned around ontology requirements.
