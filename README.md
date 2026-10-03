# research-engine

Implementation repository for a provenance-first automated research system.

The architectural contracts are developed and validated in `adamtrott78/workspace-control`
under `architecture/research-engine/`. This repository owns executable implementation.

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
