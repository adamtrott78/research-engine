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

## Docling standard/no-OCR normalization and content gate

After the stopped-daemon `standard_no_ocr` parser bakeoff has already completed,
do not rerun Docling merely to evaluate its structured output. Replay the
preserved DoclingDocument JSON through the existing `paper-package.v1`
normalizer and compare it with both Mathpix and the already-normalized
Docling-native packages:

```bash
deploy/run-docling-standard-structure-replay.sh \
  /absolute/path/to/document-docling-standard-no-ocr-OUTPUT_ROOT \
  /absolute/path/to/docling-native-normalized-root
```

The replay keeps `research-engine-daemon.service` inactive/dead, refreshes only
the core editable research-engine install, and invokes no parser runtime, model
prefetch, OCR, formula enrichment, VLM, Marker, PaddleOCR, or MinerU path.

The normalized standard package uses a mode-specific compiler identity so it
cannot collide with the validated native package identity. Evaluation reports
count agreement separately from content fidelity. Content-aware reference
metrics include:

- table text, grid-shape, and coordinate-aligned cell fidelity;
- equation representation/content agreement;
- semantic figure-to-caption association and caption text agreement;
- title/heading content plus heading-to-section-path consistency;
- block/asset page validity and bbox coverage;
- direct per-case and aggregate deltas versus normalized Docling native.

Mathpix remains a manual benchmark reference. The original PDF remains
scientific authority, including when Mathpix and a candidate parser disagree.


## Hybrid native/structure routing and targeted formula follow-up

The parser bakeoff selects a hybrid component architecture rather than one
whole-document winner. Production lexical text, reading order, and page
provenance remain sourced from the validated Docling-native package.
`standard_no_ocr` is a selective structure overlay:

- title/heading structure with valid provenance is promoted;
- picture blocks are promoted as semantic figures only when Docling links a
  non-empty caption;
- table grids are admitted only as derived structure candidates and always
  retain an authoritative-PDF verification requirement;
- standard/no-OCR equation text is not promoted after the measured zero-content
  equation result; formula-bearing pages are instead flagged for targeted
  enrichment.

Production routing is deterministic and does not consult Mathpix. Mathpix is
used only by benchmark diagnostics and promotion gates.

The stopped-daemon follow-up runner executes the four related gates in one
bounded operation:

```bash
bash deploy/run-document-hybrid-routing-followup.sh
```

An optional first argument overrides the preserved
`document-docling-standard-no-ocr-OUTPUT_ROOT`. The runner reuses the existing
normalized native and standard packages, writes the hybrid routing report,
writes a compact table residual audit, derives formula-bearing pages from the
preserved standard Docling provenance, then runs formula enrichment one page at
a time only for those targets. It does not repeat the 73-page standard parser
bakeoff.

The table residual audit is benchmark-only. It exposes candidate/reference
grids and normalized cell tokens when matched table content, shape, or aligned
cells fall below the diagnostic threshold. Those Mathpix-derived residuals may
inform engineering changes but cannot become a production routing dependency.

The formula experiment uses Docling 2.131.0's `codeformulav2` preset with the
local `docling-project/CodeFormulaV2` model pinned to
`0284bc97b73c88776957d00d6695dd63a5a8784e`. OCR, table reconstruction, code
enrichment, picture analysis, remote services, and the general document-VLM
pipeline remain disabled in that experiment. The model is fetched only into the
formula benchmark cache after the measured equation residual has justified it.
The model card currently identifies the weights as
`cdla-permissive-2.0`; deployment/redistribution obligations must be reviewed
again before production promotion.

Formula promotion is not automatic. The benchmark requires at least 0.8
reference-weighted matched-content F1, at least 0.8 reference coverage at
F1>=0.5, and non-zero content/coverage on every equation-bearing case.
The original PDF remains scientific authority throughout.


## Targeted TableFormer ACCURATE residual gate

The content-aware table audit deliberately reports both genuine extraction
failures and representational differences against the Mathpix benchmark
reference. A heavier table mode must therefore be opened only for intrinsic
failures, not every low benchmark score.

The targeted selector is intentionally narrow: only standard-no-OCR TableFormer
FAST assets whose normalized table grid is actually empty are eligible. The
selector maps each failed table asset back to its authoritative PDF page using
paper-package provenance.

Run the stopped-daemon gate with:

```bash
bash deploy/run-tableformer-accurate-residual-gate.sh
```

The gate reuses the already-installed Docling 2.131.0 standard runtime and
existing layout/TableFormer model cache. It does not rebuild the runtime or
redownload models. It converts only the selected pages using
`TableFormerMode.ACCURATE`, with OCR, formula enrichment, picture analysis,
remote services, and broad document reruns disabled.

Evaluation compares only the selected failed table references. The original PDF
remains scientific authority and Mathpix remains a manual benchmark reference.

New operator-facing temporary outputs are written beneath the visible
`~/adamArchives/Adam/workspace-control-tmp/` directory so JupyterLab can
browse/download them. Hidden `.workspace-control/tmp` paths are treated as
legacy input compatibility only.


## Targeted Paddle table-recognition fallback

TableFormer FAST and ACCURATE are considered exhausted only for table assets
that remain present in Docling provenance but contain zero structured cells in
both modes. That measured residual opens one alternate specialized component:
PaddleOCR 3.7.0 General Table Recognition v2 (PP-TableMagic).

The fallback is region-scoped, not a document parser. It renders the validated
Docling table bbox from the authoritative PDF at 300 DPI with a 6-point margin,
then supplies only that crop to `TableRecognitionPipelineV2`. Document
orientation classification, unwarping, and layout detection are disabled.
Table classification, wired/wireless structure recognition, cell detection,
and OCR remain enabled because they are the table pipeline itself.

The benchmark pins PaddleOCR 3.7.0 and PaddlePaddle 3.3.0 and records the
explicit model family in
`paddle-table-fallback-contract.v1.json`. One pipeline instance stays resident
while all selected crops are processed. Mathpix remains benchmark reference
only; the original PDF region is authoritative.

Run the stopped-daemon Lambda gate with:

```bash
bash deploy/run-paddle-table-fallback-gate.sh
```

The gate selects exactly the persistent empty-grid residuals and writes all new
crops, raw outputs, and evaluation JSON beneath the visible
`~/adamArchives/Adam/workspace-control-tmp/` root.


## Promoted persistent-empty-grid table fallback

The bounded Lambda PP-TableMagic gate recovered all three tables for which
Docling TableFormer FAST and ACCURATE both detected a table region but emitted
zero structured cells. The aggregate recovery gate passed:

- 3/3 target tables materialized;
- reference coverage at content F1 >= 0.5: 1.0;
- weighted matched-content F1: 0.8902691511387163;
- weighted aligned-cell F1: 0.8472714360469462.

This result promotes PaddleOCR 3.7.0 General Table Recognition v2 only as the
persistent-empty-grid fallback. It does not replace Docling TableFormer.

Production routing is:

```text
Docling standard_no_ocr / TableFormer FAST
  -> TableFormer ACCURATE only when the FAST grid is unusable
  -> Paddle PP-TableMagic crop only when ACCURATE still has an empty grid
  -> authoritative-PDF verification remains required
```

The two Scheirer recoveries were strong (content F1 1.0 and ~0.957). The Fadul
recovery was materially better than the zero-cell TableFormer output but still
showed row/column drift in part of the table (content F1 ~0.714, aligned-cell
F1 ~0.761). Therefore every Paddle-recovered table carries the warning
`RECOVERED_BY_PADDLE_TABLE_FALLBACK; AUTHORITATIVE_PDF_VERIFICATION_REQUIRED`.

The compact measured decision is retained in
`paddle-table-promotion-decision.v1.json`. Mathpix remains benchmark reference
only and is never consulted by production routing.


## Shared local inference concurrency gate

Validated specialized models are not production-ready merely because they are
accurate in isolated one-shot scripts. Concurrent document ingestion must not
instantiate duplicate heavyweight models per worker.

The shared local inference service therefore uses one long-lived owner process
per isolated runtime:

- Docling/CodeFormulaV2 owns the equation-only escalation queue;
- PaddleOCR/PP-TableMagic owns the persistent-empty-grid table-crop queue.

All deterministic workers in one daemon process share the same broker. Each
family queue is bounded and serialized; the two families can run concurrently
on distinct physical GPUs.

The stopped-daemon acceptance gate is:

```bash
bash deploy/run-local-inference-concurrency-gate.sh
```

It does not rebuild runtimes or download models. It reuses the validated local
runtimes/caches, submits three formula pages and three table crops concurrently,
then repeats one request from each family as a warm mixed wave. Acceptance
requires one server PID and one model load per family, one spawn per owner, no
request failures, drained queues, and sampled GPU/system-memory telemetry.

The machine-readable architecture and acceptance contract is retained in
`local-inference-service-contract.v1.json`.


## EXTRACT / VERIFY paper-package A/B

Document-parser quality does not by itself justify changing semantic research
context. The paper-package A/B gate therefore evaluates downstream behavior
against completed historical EXTRACT and VERIFY jobs.

The paper-package arm uses normalized Docling-native blocks for lexical text.
The validated standard/no-OCR package contributes structural policy signals,
currently the conservative start of the references section. EXTRACT can omit
pages strictly after that boundary; VERIFY retains the full package page set so
an authoritative cited page is never removed by context optimization.

Before any semantic replay, every historical evidence quote is checked against
the package-derived text on its original page. The default preflight requires
at least 95% of quotes to achieve token recall >= 0.8. Failure stops the gate
before any Terra/Codex calls.

The semantic sample selects low/median/high historical-input-token EXTRACT and
VERIFY jobs. Package disagreement triggers a current-PDF control only for that
job. A package-only failure is a context-regression signal; disagreement in
both arms is treated as model-variance ambiguity rather than silently blamed on
the document compiler.

Exact token usage and latency are retained separately from any quantized
ChatGPT-plan allowance observations.


## Measured localization matrix v2

`localization-matrix.v2.json` is the measured production-placement policy for
the research pipeline.

It covers every research `JobType` exactly once and is checked against
`JOB_TIERS` in CI. It also records the finalized document-component decisions:
authoritative PDF remains semantic authority; native/standard Docling stays
local document support; targeted formula/table residuals use the validated
shared local inference service; OCR/general document VLM remain disabled; and
paper-package context remains a derived cache/benchmark representation rather
than the EXTRACT/VERIFY production hydrator.

Any future proposal to retier SCREEN, EXTRACT, or VERIFY must add a controlled
quality benchmark and a material resource or allowance benefit before changing
the scheduler contract.


## Production acceptance and CodeFormulaV2 deployment review

`lambda-production-acceptance-contract.v1.json` captures the bounded Lambda
acceptance requirements after parser/routing/concurrency/localization
validation.

`codeformulav2-license-review.v1.json` records the deployment review for the
pinned CodeFormulaV2 revision. The published model metadata declares
CDLA-Permissive-2.0. The current production scope is internal inference from a
local cache; research-engine does not redistribute the weights. Any future
redistribution must preserve the applicable sharing obligation and should
trigger a fresh review.


## Bounded production integration contract

`bounded-production-integration-contract.v1.json` defines the production
sidecar/cache and finite daemon-lifecycle acceptance boundary.

The cache is keyed by authoritative source SHA-256 and emits
`document_sidecar_events` telemetry. The routing path is native lexical
package + standard/no-OCR structure, then only validated targeted residuals:
CodeFormulaV2 formula pages and FAST-empty table pages through ACCURATE before
Paddle fallback.

The integration gate fingerprints scientific tables before and after and allows
only derived cache, sidecar telemetry, and worker/ops telemetry to change.

## Residual escalation and local semantic fallback contract

The next localization stage is governed by
`residual-escalation-contract.v1.json`. It does **not** reopen the parser
bakeoff and does not globally enable OCR, a document VLM, embeddings/reranking,
or a local semantic agent.

The fixed ladder is:

```text
deterministic native + structure
  -> already-validated specialized local recovery
  -> residual-scoped OCR / multimodal / local-semantic candidate
  -> current STANDARD Codex over authoritative PDF/page/region
  -> human review
```

The contract defines eight residual families: textless/image-only scans,
low-confidence text recovery, formula residuals after CodeFormulaV2, table
residuals after FAST/ACCURATE/PP-TableMagic, semantically rich or uncaptioned
figures/charts, layout/reading-order anomalies, semantic EXTRACT ambiguity, and
semantic VERIFY ambiguity.

Every family has an oracle-free production opener, hard quality threshold,
page/region provenance requirements, fail-closed fall-through behavior,
GPU/RAM/latency ceiling, exact semantic-token accounting, and a required
contemporaneous authoritative-PDF Codex control before promotion.

The minimum corpus expansion is intentionally small: three new source PDFs are
required to add (1) mixed-quality scanned pages, (2) quantitative/relational
visual semantics, and (3) a malformed-text-layer or reading-order anomaly.
Formula/table residuals reuse the existing seven-paper corpus, and semantic
EXTRACT/VERIFY ambiguity reuses preserved authoritative source artifacts plus
calibration evidence. Large source PDFs and model weights remain outside Git.

Concrete new OCR, multimodal, and local-agent models are deliberately not named
or installed by this first contract checkpoint. Candidate selection opens only
after the contract is durable.

## Residual fallback candidate selection

After `residual-escalation-contract.v1.json` became durable, the initial
benchmark-only candidate set was reduced to three primary families in
`residual-candidate-selection.v1.json`:

- `PP-OCRv6_medium` for page-scoped scan/text and layout recovery;
- `PaddleOCR-VL-1.6` for document-specific figure/chart and exhausted
  formula/table/layout regions;
- `Qwen3.5-4B` for bounded local semantic EXTRACT/VERIFY experiments and,
  only if the smaller document VLM fails or abstains, general visual fallback.

Selection is not promotion. No weights/runtime are downloaded by the selection
checkpoint, no production route or job tier changes, and the research daemon
lifecycle remains unchanged. The three missing source-case manifests must be
durable before bounded runtime prefetch begins.

The candidate sequence is intentionally asymmetric: purpose-built OCR is tried
before a VLM; the compact document VLM is tried before the general multimodal
model; and the general model is never run on every page. Already-promoted
CodeFormulaV2 and FAST -> ACCURATE -> PP-TableMagic paths remain unchanged.

## Residual source-case registration

The three missing source cases required by
`residual-escalation-contract.v1.json` are registered under `cases/` before
any new model/runtime prefetch:

- `scan-mixed-quality-v1.json`: NASA TN D-486, with one image-dominant scan
  candidate and one degraded textual scan candidate;
- `semantic-visuals-v1.json`: *Deep Residual Learning for Image Recognition*,
  targeting a quantitative training-error chart and a relational architecture
  diagram;
- `malformed-layout-v1.json`: *Faster R-CNN*, targeting complex two-column
  pages with full-width figures/footnotes as native reading-order stress cases.

Registration does not claim that the public web crawl is the authoritative
binary or that an opener has fired. The bounded Lambda source-prefetch/audit
gate must download the original PDFs, capture SHA-256 and page counts, and run
the same deterministic native/structure diagnostics used by production. The
layout source is fail-closed: if no target page actually satisfies the
governing opener, it is replaced before any layout model inference.

Large PDFs, rendered crops, and future model weights stay outside Git. These
manifests retain only stable source locators and compact benchmark metadata.

## Bounded residual source prefetch/audit

The three registered residual source cases are validated by
`residual-source-audit-contract.v1.json` and
`deploy/run-residual-source-prefetch-audit.sh` **before** any newly selected
OCR/VLM/local-semantic weights are installed.

This gate downloads only the three registered PDFs into the visible
`~/adamArchives/Adam/workspace-control-tmp/` tree, records SHA-256/page counts,
then runs the already-installed Docling native and `standard_no_ocr` paths.
The structure pass is forced offline against the existing validated model
cache.

The deterministic audit evaluates the durable scan/text/layout openers and
locates the two registered semantic-visual figure regions through
caption-linked PDF provenance. It performs zero candidate-model downloads,
zero candidate-model inference calls, zero semantic calls, and no scientific
state materialization.

The gate is fail-closed. In particular, `malformed-layout-v1` is rejected and
must be replaced if neither registered target page actually fires the layout
opener; thresholds are not relaxed to preserve the chosen source. Model
prefetch remains closed unless both semantic visual targets resolve to unique
caption-linked figure bboxes as well.

Run only from the clean canonical `main` checkout with the research daemon
stopped:

```bash
bash deploy/run-residual-source-prefetch-audit.sh
```

All bulky source PDFs, parser outputs, packages, and audit reports stay under
the visible temporary root rather than Git.

## Residual source gate PASS and bounded model prefetch

The completed Lambda source audit is compactly pinned in
`residual-source-audit-pass.v1.json`. All three source families passed:
scan/text openers are present, both semantic visual targets resolve to unique
caption-linked PDF bboxes, and both registered malformed-layout target pages
fire the durable layout opener. The original PDFs remain authority; the full
high-volume audit report remains outside Git.

That PASS opens only the next bounded gate:
`residual-model-prefetch-contract.v1.json` with
`deploy/run-residual-model-prefetch-gate.sh`.

The model-prefetch gate does **not** run OCR, VLM, or semantic inference. It:

- resolves and downloads the PP-OCRv6 medium detection/recognition artifacts;
- downloads PaddleOCR-VL-1.6 and Qwen3.5-4B at their already-pinned revisions;
- records exact resolved revisions, Apache-2.0 license metadata, per-file
  SHA-256 manifests, materialized byte counts, and offline reopenability;
- pins the PaddleOCR 3.7.0 wheel without installing its inference stack; and
- records host/GPU compatibility facts using only the already-validated
  standard Docling/Torch runtime.

Candidate inference runtimes such as PaddlePaddle, vLLM, SGLang, or
Transformers are intentionally **not** installed by this gate. After the
prefetch receipts are durable, isolated inference runtimes and residual-scoped
quality/resource benchmarks open in the fixed order: PP-OCRv6 scan/text,
PaddleOCR-VL targeted document/visual residuals, then Qwen3.5-4B bounded
semantic controls. Qwen visual use remains conditional on PaddleOCR-VL
failure or abstention.

## PP-OCRv6 bounded residual benchmark

After the model-prefetch gate passes, the first candidate inference gate is
`ppocrv6-residual-benchmark-contract.v1.json`, executed by
`deploy/run-ppocrv6-residual-benchmark.sh`.

The gate constructs one isolated PaddlePaddle/PaddleOCR runtime, verifies the
prefetched PP-OCRv6 medium detection and recognition weight checksums, renders
only the two promoted pages from the authoritative NASA scan PDF, and runs
exactly three OCR calls: one first/cold page, one warm second target, and one
warm repeat.

The benchmark uses only local PP-OCRv6 model paths and disables document
orientation, document unwarping, and textline-orientation modules. It records
model initialization time, warm-page latency, process GPU memory, peak RSS,
recognized lines/scores/boxes, exact page provenance, and an isolated runtime
package freeze.

This gate is measurement-only. Resource success produces
`MEASURED_PENDING_HUMAN_QUALITY`, not promotion. The two rendered page PNGs
and compact OCR report must be reviewed against visible authoritative page
content before PP-OCRv6 can suppress a higher tier for scan/text residuals.
No VLM, Qwen, Codex, production routing, or scientific-state mutation is
allowed in this gate.

## PP-OCRv6 measured outcome and PaddleOCR-VL visual gate

The bounded PP-OCRv6 Lambda run passed every runtime/resource ceiling but did
not earn production promotion.

The model initialized in under one second, remained below 2 GiB process VRAM
and 1.5 GiB peak RSS, and kept the worst warm page below 0.34 seconds. Human
review nevertheless found two independent benchmark-quality issues:

- the selected image-only/textless NASA page was visually almost blank, so a
  correct zero-text OCR abstention does not demonstrate recovery quality for a
  content-bearing scanned page; that scan benchmark target must be replaced;
- the low-confidence chart was recovered substantially better than its native
  text layer, but small right-margin labels were missed/fused and the vertical
  HEIGHT label was truncated, so the hard residual quality thresholds cannot
  be claimed.

The compact fail-closed review is retained in
`ppocrv6-human-quality-review.v1.json`. PP-OCRv6 remains measured evidence,
not a promoted production recovery path.

That result opens the next already-selected rung only for visual semantics:
`paddleocr-vl-residual-benchmark-contract.v1.json` with
`deploy/run-paddleocr-vl-residual-benchmark.sh`.

The PaddleOCR-VL gate uses the already-prefetched PaddleOCR-VL-1.6 revision
`c5630abae1d940eafe0697512a0325494b02ab42`. It renders only the two
caption-linked figure bboxes accepted by the residual source audit, disables
layout detection, document orientation, and unwarping, and runs the VLM with
the Transformers engine against local model files. No PP-DocLayoutV3,
Qwen3.5-4B, Codex, whole-document VLM parsing, daemon start, production route
change, or scientific-state mutation is allowed.

A successful local VLM resource run remains
`MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL`. Human annotated-fact review must
pass first; only then may the required contemporaneous authoritative-PDF Codex
control open before any promotion decision.

### PaddleOCR-VL runtime dependency correction

The first Lambda initialization attempt reached the pinned local
PaddleOCR-VL-1.6 model but performed zero inference calls because its custom
Transformers model code imports `torchvision`, which was absent from the
initial isolated runtime. The model also requested explicit approval to execute
its pinned custom code.

The corrected runtime pins `torch==2.8.0` with matching
`torchvision==0.23.0` from the CUDA 12.6 PyTorch wheel index and passes
`engine_config={"trust_remote_code": True}` only for the already-prefetched,
checksum-verified PaddleOCR-VL-1.6 local model directory. A runtime-ready
sentinel is now invalidated and the venv rebuilt when the required Torch,
TorchVision, or Transformers versions are missing. The failed initialization
had zero candidate inference calls and therefore contributes no quality result.

### PaddleOCR-VL protobuf dependency correction

The second Lambda initialization attempt again performed zero candidate
inference calls. The rendered Figure 6 and Figure 3 crops were correct, but
Transformers stopped during local custom-model initialization because the clean
runtime did not contain the optional `protobuf` package.

The corrected runtime pins `protobuf==5.29.5` and includes its version in the
runtime-ready sentinel contract. Any prior ready environment without the exact
protobuf dependency is invalidated and rebuilt before inference. This change
does not alter the model, the two benchmark regions, inference count, resource
ceilings, quality thresholds, daemon policy, or promotion boundary.

### PaddleOCR-VL direct Transformers loader correction

The third Lambda initialization attempt again executed zero candidate
inference calls, but this time all required Python packages were present.
Transformers successfully loaded the pinned custom
`PaddleOCRVLConfig`, then PaddleX 3.7 requested
`AutoModelForImageTextToText`. The pinned model's own `config.json`
does not declare that auto-class; it declares `AutoModel` and
`AutoModelForCausalLM`. Transformers 4.55.0 therefore rejected the
PaddleX wrapper before weights were initialized.

This is an interface mismatch, not another missing dependency and not a
PaddleOCR-VL quality failure.

The benchmark now bypasses only the incompatible PaddleX Transformers wrapper.
It keeps the exact same prefetched model directory and weight checksum, but
loads through the model repository's declared local custom-code interfaces:
`AutoModelForCausalLM` and `AutoProcessor`, with
`trust_remote_code=True`, `local_files_only=True`, and offline environment
flags. No model file is modified.

The two benchmark inputs and three-call budget are unchanged. Figure 6 uses the
model's documented `Chart Recognition:` element-level prompt; Figure 3 uses
the documented `OCR:` prompt. If those model-native tasks cannot recover the
required visual facts/relations, that is a legitimate capability result and
must fall through to the selected Qwen visual fallback rather than trigger
another wrapper patch.

### PaddleOCR-VL causal-mask keyword compatibility

The fourth Lambda attempt loaded substantially farther into the direct
Transformers path and allocated roughly 2.34 GiB of process VRAM, but still
executed zero candidate inference calls. The failure was:

`TypeError: create_causal_mask() got an unexpected keyword argument 'inputs_embeds'`.

This is a known API mismatch between the pinned PaddleOCR-VL-1.6 custom model
code and Transformers 4.55.0. The model calls
`transformers.masking_utils.create_causal_mask(..., inputs_embeds=...)`, while
Transformers 4.55.0 exposes the singular parameter `input_embeds`.

The benchmark now installs a narrow in-memory compatibility alias before the
custom modeling module loads. It inspects the installed function signature,
fails closed if neither expected spelling is present, and otherwise translates
only `inputs_embeds` to `input_embeds`. No model file, model weight,
Transformers package, prompt, benchmark region, inference count, quality
threshold, or production behavior is changed.

### PaddleOCR-VL generation bound correction

The first causal-mask-compatible direct-model run reached actual VLM generation
but remained inside the first call for more than five minutes. This is not a
setup/dependency stall: the model had already passed local processor/model
preflight and entered generation.

The harness had allowed `max_new_tokens=2048` while explicitly using
`use_cache=False`. PaddleOCR-VL-1.6's own generation configuration also
declares `use_cache=false`, and its documented direct-Transformers example
uses a 512-token generation cap. The bounded benchmark now matches that
documented 512-token cap instead of permitting four times as much no-cache
decoding.

Each call now emits a flushed `VLM_CALL_START` event before generation and a
`VLM_CALL_COMPLETE` event with elapsed seconds afterward so long-running
generation is never silent. Model files, weights, prompts, crops, inference
count, quality thresholds, resource ceilings, and production boundaries remain
unchanged.
