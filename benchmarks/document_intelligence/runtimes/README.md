# Isolated document-parser runtimes

The research engine intentionally does **not** install Docling, MinerU,
PaddleOCR, Marker, Torch, PaddlePaddle, vLLM, or model weights into its core
runtime.

Each candidate is benchmarked from a separate environment and is discovered
through an explicit executable override. This avoids dependency collisions and
makes candidate removal/upgrade independent of the research daemon.

The pinned benchmark contracts live in:

```text
benchmarks/document_intelligence/candidate-runtime-contracts.v1.json
```

## Status

With no candidate runtimes installed:

```bash
research-engine document-adapter-status
```

returns each candidate as unavailable. That is expected and is a fail-closed
state.

A runtime is considered runnable only when:

1. its executable exists;
2. the corresponding Python distribution version can be verified from the same
   environment;
3. that version matches the pinned benchmark contract.

## Suggested environment layout

On a development machine or, later, the Lambda:

```text
~/adamArchives/Adam/.document-runtimes/
  docling/
  mineru/
  paddle/
  marker/
```

These paths are a convention only. The benchmark does not create them.

### Docling 2.131.0

Code license: MIT.

```bash
python3 -m venv ~/.document-runtimes/docling
~/.document-runtimes/docling/bin/python -m pip install -U pip

# Cheap native-PDF phase: CLI + PDF parser, no local-model/Torch bundle.
~/.document-runtimes/docling/bin/python -m pip install \
  'docling-slim[cli,format-pdf]==2.131.0'

export RESEARCH_ENGINE_DOCLING_BIN=~/.document-runtimes/docling/bin/docling
```

Benchmark modes:

- `native`: PDF-native parser, no model inference; the lightweight install
  above is sufficient;
- `standard`: standard document pipeline; promote the same environment with
  `pip install 'docling==2.131.0'`;
- `vlm`: VLM document pipeline; promote with
  `pip install 'docling[vlm]==2.131.0'`.

The adapter probes the `docling-slim` distribution version, so both the slim
and full environments satisfy the same 2.131.0 contract. It requests both
Markdown and structured Docling JSON.

### MinerU 4.x

MinerU uses the MinerU Open Source License: Apache-2.0 plus additional
commercial-threshold and online-service attribution terms. Review the current
license before any deployment outside this research environment.

```bash
python3 -m venv ~/.document-runtimes/mineru
~/.document-runtimes/mineru/bin/python -m pip install -U pip
~/.document-runtimes/mineru/bin/python -m pip install 'mineru>=4.0,<5'

export RESEARCH_ENGINE_MINERU_BIN=~/.document-runtimes/mineru/bin/mineru-kit
```

Benchmark modes:

- `flash_native`: native PDF text, `--tier flash --ocr-mode txt`, no inference;
- `basic`: small-model parsing;
- `standard`: normal high-quality local parsing;
- `advanced`: highest local inference tier.

The benchmark never adds `--remote`.

### PaddleOCR / PP-StructureV3 3.7.x

Code license: Apache-2.0. Model cards/weights must still be reviewed for the
specific selected component set.

PaddlePaddle itself must be installed using the hardware-appropriate package
for the target host. After that:

```bash
python -m pip install 'paddleocr[doc-parser]>=3.7,<3.8'

export RESEARCH_ENGINE_PADDLE_PYTHON=/path/to/paddle/venv/bin/python
```

The research-engine adapter calls the standalone shim:

```text
benchmarks/document_intelligence/runtimes/paddle_ppstructurev3.py
```

The shim uses PP-StructureV3's documented Python API and emits:

```text
document.md
pages.json
images/...
```

The initial English-paper benchmark selects `en_PP-OCRv4_mobile_rec`, as
recommended by the PP-StructureV3 documentation for English-only documents.

### Marker 2.0.0

Marker code: Apache-2.0.

Marker model weights: modified AI Pubs Open Rail-M with separate commercial
conditions. Do not treat the code license as the weight license.

Even `fast_no_ocr` lives inside an environment whose Marker 2.0 package
directly depends on Torch, Transformers, and Surya. Treat that as a heavy
installation footprint even though the selected conversion mode avoids VLM
inference.

```bash
python3 -m venv ~/.document-runtimes/marker
~/.document-runtimes/marker/bin/python -m pip install -U pip
~/.document-runtimes/marker/bin/python -m pip install 'marker-pdf==2.0.0'

export RESEARCH_ENGINE_MARKER_BIN=~/.document-runtimes/marker/bin/marker_single
```

Benchmark modes:

- `fast_no_ocr`: `fast --disable_ocr`, CPU/text-layer oriented;
- `fast`: selective local inference;
- `balanced`: higher-quality VLM-assisted local mode.

The benchmark never enables Marker's `--use_llm` option.

## Running one candidate

Check the runtime first:

```bash
research-engine document-adapter-status
```

Run one PDF:

```bash
research-engine document-adapter-run \
  docling standard paper.pdf \
  --output-dir /tmp/docling-paper
```

Run the full seed:

```bash
research-engine document-benchmark-external \
  benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json \
  docling standard \
  --bundle-root /path/to/unpacked-benchmark \
  --output-root /tmp/document-bakeoff
```

External outputs are initially marked `RAW_EXTERNAL_OUTPUT`. They receive the
same Mathpix lexical/order/resource metrics, but **do not** claim
`paper-package.v1` provenance until a parser-specific structured-output
normalizer is implemented and validated.

## Local-only execution

Candidate subprocesses run with remote provider secrets stripped from their
environment, including OpenAI, Anthropic, Google/Gemini, OpenRouter, Mathpix,
and MinerU API credentials.

This is defense in depth. Candidate command templates also exclude:

- MinerU `--remote`;
- Marker `--use_llm`;
- any Mathpix/API conversion;
- any research-engine scientific-state mutation.

The benchmark runner invokes argv directly without a shell.

## Lambda gate

Do not install these environments on the Lambda merely because this
documentation exists.

The durable task capsule controls when candidate installation is allowed.
Initial development and command-contract validation happen in GitHub/CI first.
Only selected candidates advance to Lambda hardware/resource benchmarking.
