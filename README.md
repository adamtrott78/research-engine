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
