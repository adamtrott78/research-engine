# Document intelligence benchmark

This directory is the development and comparison surface for
`research-engine-local-document-pipeline-v1`.

The benchmark is intentionally separate from production scientific state.
Running an adapter here must not create evidence objects, verification events,
gaps, checkpoints, or human-review decisions.

## Authority

The original content-addressed PDF is authoritative.

Every parser output, including Mathpix Markdown, is a derived representation.
A benchmark may compare derived outputs against Mathpix, but a disagreement
between Mathpix and the PDF is resolved in favor of the PDF.

## paper-package.v1

The Python contract lives in:

```text
src/research_engine/document_package.py
```

A compiled package is expected to expose a directory shaped approximately as:

```text
paper-package.v1/
  manifest.json
  document.md
  blocks.json
  diagnostics.json

  tables/
    table-001.md
    table-001.csv
    table-001.json

  figures/
    figure-001.png
    figure-001.json

  equations/
    equation-001.json
```

The exact on-disk writer is not implemented yet. The contract deliberately
keeps source page/bbox/section provenance in structured records rather than
encoding scientific authority into Markdown text.

Package cache identity is derived from:

1. authoritative source PDF SHA-256;
2. compiler/adapter identity;
3. compiler version;
4. canonical compiler configuration hash.

Changing any of those inputs creates a new derived package identity.

## Benchmark inputs

Do **not** commit large source PDFs or model weights here.

A benchmark case manifest records the source PDF SHA-256 and a source location
that can refer to the Lambda content-addressed artifact store, a user-supplied
file, or another explicitly materialized test location.

Representative cases should cover at least:

- born-digital single-column text;
- born-digital multi-column scientific papers;
- malformed or garbled embedded text;
- scanned pages;
- equation-heavy pages;
- simple and merged/complex tables;
- figures with captions;
- quantitative charts;
- diagrams;
- mixed text/table/figure pages;
- long reference sections;
- headers/footers/page-number artifacts.

## Mathpix oracle workflow

Mathpix is an **optional human-generated benchmark oracle**, not a production
dependency.

The user can use the desktop Mathpix workflow under their non-API annual page
allowance, then provide or push the generated Markdown/image bundle for a
benchmark case.

Recommended Git layout for small oracle outputs:

```text
benchmarks/document_intelligence/oracles/<case-id>/
  document.md
  oracle-manifest.json
  images/                 # only when reasonably small
```

For large image bundles, keep the binary assets outside Git and commit only the
oracle manifest plus Markdown. The manifest should record paths/hashes to the
external assets.

Never add a Mathpix API key or unattended paid Mathpix API call to the research
engine.

## Initial MILCOM/Mathpix seed

The first benchmark seed is:

```text
benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json
```

It contains seven manually selected PADataset papers / 73 PDF pages spanning
long-form open-set/ML papers and short IEEE/MILCOM RF-security papers. The seed
includes equation-heavy, table-heavy, figure-heavy, algorithmic, and
multi-column cases.

Mathpix filename extensions are not used to infer asset behavior. Each case
records the observed oracle image mode based on the actual markup:
`REMOTE_IMAGES`, `LOCAL_IMAGES`, `MIXED`, or `NO_IMAGES`.

After unpacking the user-supplied benchmark bundle so the root contains
`papers/`, run the deterministic baseline with:

```bash
research-engine document-benchmark-native \
  benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json \
  --bundle-root /path/to/unpacked-bundle-root \
  --output-root /tmp/document-benchmark-native
```

The native baseline uses pypdf's text layer only and runs both `plain` and
`layout` extraction modes by default. Each mode emits one page-grounded block
per source page and writes independent `paper-package.v1`
manifest/block/diagnostic files. The two native modes are measured separately
because fixed-width layout reconstruction is not assumed to improve semantic
reading order.

Neither mode claims to reconstruct tables, equations, figures, or fine-grained
document structure. Their role is to establish the cheapest possible baselines
that every model-backed adapter must beat. Use `--modes plain` or
`--modes layout` to run only one variant.

The Mathpix comparison currently reports a coarse lexical multiset F1 and
ordered-bigram F1. These are baseline diagnostics, not acceptance metrics by
themselves; later structured-content and downstream EXTRACT/VERIFY metrics are
the promotion gates.

## Candidate adapters

Initial candidates are intentionally component-oriented:

- native PDF text/geometry/image extraction;
- Docling;
- MinerU;
- PaddleOCR / PP-StructureV3;
- Marker / Surya;
- deterministic OCR/table/formula components where useful;
- hybrid adapters composed from measured component winners.

Pinned versions, licenses, mode commands, capability metadata, and isolated
runtime requirements are recorded in:

```text
candidate-runtime-contracts.v1.json
runtimes/README.md
```

The research-engine core does not install those heavyweight runtimes. Inspect
what is available with:

```bash
research-engine document-adapter-status
```

A single installed candidate can be exercised with
`research-engine document-adapter-run`, and the full seed with
`research-engine document-benchmark-external`. Candidate subprocesses run
without a shell and with remote LLM/OCR API credentials removed from their
environment.

Until a parser-specific structured-output normalizer proves page/block
provenance, external results are marked `RAW_EXTERNAL_OUTPUT` and have no
`paper-package.v1` package ID. This prevents a visually good Markdown export
from being mistaken for provenance-safe research input.

No candidate is accepted based on Markdown appearance alone.

## Metrics

Document-level quality metrics:

- text fidelity;
- reading order;
- equation/LaTeX fidelity;
- table structure/cell fidelity;
- figure-caption association;
- page/bbox provenance accuracy;
- warnings/failure rate.

Resource and throughput metrics:

- wall time;
- pages/second;
- peak RSS;
- peak VRAM;
- mean CPU/GPU utilization where available;
- model-load/cold-start time;
- sustained throughput under concurrent PDFs;
- queue/backpressure behavior.

Downstream research metrics:

- Markdown/context character and token count;
- EXTRACT evidence agreement;
- VERIFY outcome agreement;
- accepted evidence-ID agreement;
- semantic input/cached/output/reasoning tokens;
- local compute time and resource cost;
- censored plan-allowance movement only as a separate empirical observation.

## Routing policy

The intended architecture is a cascade:

```text
native PDF extraction
        |
        +-- healthy text/layout ----------> deterministic normalization
        |
        +-- OCR/layout/table/math needed -> specialized local model
        |
        +-- low confidence/complex region -> document VLM
        |
        +-- unresolved evidence-critical region
                                      ----> remote/human escalation
```

A general local VLM must not be invoked on every page by default.

Figures are extracted and caption-linked eagerly, but semantic figure/chart
interpretation is scheduled only when downstream work actually needs it or
parsing confidence is low.

## Concurrency

Production validation must measure multiple PDFs in flight. Heavyweight models
should be shared through a serving/batching layer or equivalent bounded worker
pool instead of loading duplicate model copies per document worker.

Single-document latency is not sufficient evidence for production adoption.

## GitHub result exchange

During development, adapters may commit small Markdown/JSON benchmark outputs
and metrics so ChatGPT can inspect them directly through GitHub.

Git is not the production artifact transport. High-volume compiled packages
belong in the research-engine artifact/cache store.

## Lambda gate

A parser or local model is not production-ready merely because it works in CI
or on a desktop.

Before integration it must reproduce on the Lambda and report:

- exact dependency/model versions;
- model/weight cache location;
- CPU/RAM/GPU/VRAM use;
- cold and warm performance;
- concurrent document throughput;
- failure/restart behavior;
- final package/provenance validation.

The research daemon remains stopped during early benchmark work.
