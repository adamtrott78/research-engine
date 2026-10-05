# research-engine

Implementation repository for a provenance-first automated research system.

The architectural contracts are developed and validated in `adamtrott78/workspace-control`
under `architecture/research-engine/`. This repository owns executable implementation.

## Workspace-control scope binding

Substantial multi-step work in this repository is governed by the shared
workspace-control Scope Closure Gate pinned by
`policies/workspace-control-scope-adapter.v1.json`.

The adapter currently binds to:

`adamtrott78/workspace-control@1295477546c802029d7c52755beeacdef31b5346`

and routes executors to the shared:

- `procedures/scope_closure.md`;
- `schemas/scope_envelope.v1.schema.json`;
- `scripts/scope_control.py`.

Research-engine does not vendor or import that control-plane implementation.
At task start, derive the scope envelope from the authoritative owning
workspace-control task capsule, including its task ID, objective/current bounded
goal, exact `next_safe_action`, and any explicit scope authorizations.

Newly discovered work may continue immediately only when it is required for
correctness/safety/durability, directly blocks the recorded next safe action or
required validation, or is explicitly authorized. Non-blocking adjacent
hardening, refactors, new model comparisons, extra experiments, and neighboring
runner improvements are deferred as follow-up work.

Once the owning objective is durable, required validation passes, and no
execute-now finding remains, the required decision is
`CHECKPOINT_AND_STOP`.

## v1 invariants

- scientific/source records are authoritative;
- source bytes are content-addressed by SHA-256;
- graph/vector/text indexes are rebuildable projections;
- source-derived evidence is distinct from model inference;
- verification and supersession are preserved rather than overwritten;
- no gap becomes novel merely because search failed to find prior work;
- no research program becomes executable without its required human gate.

## Initial stack

- Python 3.11+
- PostgreSQL 18
- pgvector 0.8.6
- content-addressed filesystem artifact store initially
- ChatGPT-authenticated Codex CLI as the default semantic executor

## Semantic execution

The production default is `RESEARCH_ENGINE_SEMANTIC_BACKEND=codex`.

Codex must be signed in with the operator's ChatGPT account on the host. The
research engine invokes `codex exec` non-interactively with structured output,
read-only sandboxing, and native web search for discovery/novelty work. This
uses the ChatGPT/Codex plan allowance rather than an `OPENAI_API_KEY`.

Tier defaults:

- CHEAP -> `gpt-5.6-luna`, low effort
- STANDARD -> `gpt-5.6-terra`, medium effort
- STRONG -> `gpt-5.6-sol`, high effort

The prior Responses-API executor remains available only through the explicit
development override `RESEARCH_ENGINE_SEMANTIC_BACKEND=openai_api`; production
deployment must not set that override or provide an API key.

For a host deployment, keep PostgreSQL persistent via `docker compose up -d db`
and run the daemon natively as the signed-in user (for example with the checked-in
user systemd service), so Codex can use that user's existing ChatGPT login
credentials without copying credentials into a container.

## Development

```bash
python -m pytest
```

PostgreSQL integration uses the official pgvector image tag `pgvector/pgvector:0.8.6-pg18`.

## Continuous integration

GitHub Actions runs the unit suite plus the PostgreSQL/pgvector integration test against a real PostgreSQL 18 + pgvector service. Local Docker is optional for development; CI is the canonical portable runtime check for the v1 database boundary.


### Table escalation policy

Document compilation keeps table inference cascaded and region-scoped:

1. Docling standard/no-OCR TableFormer FAST is the normal structure tier.
2. An intrinsically unusable FAST grid opens a targeted TableFormer ACCURATE
   retry for that table/page only.
3. If ACCURATE still emits an empty grid, the validated PDF table bbox is sent
   to the promoted PaddleOCR General Table Recognition v2 crop fallback.
4. Paddle-recovered tables remain derived evidence and require authoritative
   PDF verification.

The Paddle fallback passed its three-target Lambda benchmark but is deliberately
not a default document parser. Full PP-StructureV3, Marker, MinerU, and broad
document-VLM parsing remain outside this routing path.


### Shared local document inference

The validated equation and persistent-empty-grid table escalations are served
through a shared process-local broker rather than by loading heavyweight models
inside each document worker.

When explicitly enabled with
`RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1`, the daemon owns one long-lived
CodeFormulaV2 subprocess and one long-lived PP-TableMagic subprocess. All
deterministic workers receive the same broker object.

Each model family has a bounded FIFO queue and executes serially within its own
isolated runtime. Formula and table requests may execute concurrently when
bound to distinct GPUs. This preserves dependency isolation while preventing
duplicate model loads across concurrent PDFs.

The service is opt-in until the stopped-daemon Lambda concurrency/resource gate
passes. The broker health snapshot reports process PIDs, spawn counts, queue
depths, accepted/completed/failed requests, and the model-load metadata returned
by each server.


### EXTRACT / VERIFY paper-package A/B

Paper-package context is not allowed to replace the current PDF hydrator until a
non-materializing replay demonstrates source-grounded agreement.

The A/B path keeps Docling native as the lexical substrate and uses
standard/no-OCR structure only for conservative context assembly. EXTRACT may
exclude pages strictly after the detected reference-section start. VERIFY never
applies that trimming policy; exact cited pages remain available fail-closed.

The acceptance sequence is:

1. resolve the preserved nine-paper research run and authoritative source
   artifacts;
2. compile native and standard/no-OCR paper packages using the already
   validated local runtimes;
3. run a zero-model historical-quote/page-recovery preflight across all
   completed EXTRACT and VERIFY jobs;
4. only after that preflight passes, replay low/median/high historical-cost
   EXTRACT and VERIFY jobs against paper-package context;
5. if a package replay disagrees with history, run the current PDF context as a
   control for that job to distinguish context regression from model variance.

Semantic replays call the delegate directly and never invoke
SemanticResultMaterializer. They may record operational calibration events but
must not mutate scientific evidence or verification state.


### Measured localization policy

The production placement policy is captured in
`benchmarks/document_intelligence/localization-matrix.v2.json`.

The v2 matrix is evidence-backed rather than aspirational. It preserves the
scheduler's current job tiers and explicitly records where local processing is
validated, where remote/web/human reasoning is retained, and where candidate
localization was rejected or deferred.

In particular, EXTRACT and VERIFY remain STANDARD semantic jobs over
authoritative-PDF context. The measured paper-package A/B did not justify
replacing that baseline. Specialized local document models remain targeted
sidecar/escalation components rather than a silent semantic-context substitute.


### Lambda production acceptance

The local document pipeline is not enabled for sustained research merely because
its component benchmarks pass. Production acceptance is a separate bounded gate.

The production env enables one shared local-inference broker per daemon with
one formula owner on GPU 0 and one table owner on GPU 1. The live env can be
updated idempotently with a backup using
`deploy/configure-production-local-inference.sh`.

`deploy/run-lambda-production-acceptance.sh` then validates the installed
systemd unit, Codex-only production env, PostgreSQL quiescence, document
runtimes/model caches, GPU allocation, measured localization policy, internal
CodeFormulaV2 deployment-license review, zero-spawn broker construction, and a
read-only authoritative-PDF hydration smoke test.

The acceptance runner never starts the research daemon, never performs a
semantic call, never spawns the heavyweight model servers, and fails closed if
a RUNNING research run or QUEUED/RUNNING non-human job is present.


### Derived document sidecars

The production document pipeline can build a content-addressed **derived
sidecar** after authoritative ACQUIRE. The sidecar is explicitly non-authoritative:
the source PDF is committed first, EXTRACT/VERIFY continue to use
`ResearchContextHydrator` over that PDF, and a sidecar failure does not fail
ACQUIRE.

When `RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED=1`, the deterministic worker
caches native and standard/no-OCR paper packages by source SHA-256. Optional
specialized routing preserves the validated cascade: formula pages use the
shared CodeFormulaV2 owner; FAST empty table grids receive targeted TableFormer
ACCURATE; Paddle crop fallback is used only if the ACCURATE page still has no
non-empty table grid.

Operational receipts are written to `document_sidecar_events`. Cache hits
perform zero new specialized-model requests.
