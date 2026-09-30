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

## Development

```bash
python -m pytest
```

PostgreSQL integration uses the official pgvector image tag `pgvector/pgvector:0.8.6-pg18`.
