# 24/7 deployment

The daemon is a replaceable supervisor over PostgreSQL-backed durable research
state. Restarting or replacing the process does not replace scientific authority.

## Production path: ChatGPT-authenticated Codex

Production semantic execution uses the host's existing Codex CLI login and the
operator's included ChatGPT/Codex plan allowance. It must not require
`OPENAI_API_KEY` and must not silently fall back to paid API billing.

The production split is deliberate:

- PostgreSQL runs persistently under Docker Compose.
- The research daemon runs natively as the signed-in user.
- Codex CLI credentials remain on the host in the user's normal Codex config.
- The daemon calls Codex through an absolute `RESEARCH_ENGINE_CODEX_BIN` path.
- Durable artifacts live under
  `~/adamArchives/Adam/.research-engine/artifacts`.

### 1. Start PostgreSQL only

From the repository root:

```bash
sudo docker compose -f docker-compose.yml up -d db
```

The base Compose file owns only the PostgreSQL service and its persistent volume.

### 2. Install the native daemon in a dedicated venv

```bash
state="$HOME/adamArchives/Adam/.research-engine"
python3 -m venv "$state/venv"
"$state/venv/bin/python" -m pip install --upgrade pip
"$state/venv/bin/pip" install -e "$HOME/adamArchives/Adam/research-engine"
mkdir -p "$state/artifacts"
```

### 3. Create the non-secret host environment

Copy `deploy/research-engine.env.example` to:

`~/adamArchives/Adam/.research-engine/research-engine.env`

Set `RESEARCH_ENGINE_CODEX_BIN` to the absolute path returned by
`command -v codex`. The file contains no API secret.

The conservative defaults are intentional for first acceptance: one semantic
worker per tier and one active research run. Tune only after observing real
Codex allowance consumption and quality.

### 4. Install the user systemd service

```bash
install -D -m 0644   deploy/systemd/research-engine-daemon.service   "$HOME/.config/systemd/user/research-engine-daemon.service"

systemctl --user daemon-reload
systemctl --user enable --now research-engine-daemon.service
```

For unattended operation after logout, ensure user lingering is enabled on the
host:

```bash
sudo loginctl enable-linger "$USER"
```

Useful commands:

```bash
systemctl --user status research-engine-daemon.service
journalctl --user -u research-engine-daemon.service -n 100 --no-pager
"$HOME/adamArchives/Adam/.research-engine/venv/bin/research-engine" overnight-report --hours 12
```

Submit work to the already-running daemon with `submit-run`; do not start a
second daemon just to enqueue a research run:

```bash
research-engine submit-run "open-set RF research topic" \
  --run-id "rf-expansion-v1" \
  --policy-json '{"fanout":{"max_gap_candidates":2,"max_novelty_queries_per_gap":2}}' \
  --discovery-json '{"candidate_limit":20,"source_scope":"primary open-access arXiv papers"}'
```

Resolve a durable human gate explicitly:

```bash
research-engine human-review-resolve human-review:rf-expansion-v1 \
  --approve \
  --decision-json '{"reviewer":"human","decision":"approved after provenance audit"}'
```

Discovery output is deterministically capped when `candidate_limit` is supplied.
SCREEN may select only IDs that came from the bounded DISCOVERY result; the
original discovery metadata remains authoritative. Until more deterministic
acquisition adapters are installed, retained non-arXiv sources are recorded as
cancelled acquisition jobs rather than failing the research run.

## Explicit API-development profile only

`deploy/docker-compose.daemon.yml` is retained solely for explicit
API-development/testing. Its daemon service is behind the
`openai-api-development` Compose profile and sets
`RESEARCH_ENGINE_SEMANTIC_BACKEND=openai_api`.

Do not use that profile for production operation under the ChatGPT-plan
constraint.

## Operator shortcuts

Install the repository-owned operator commands:

```bash
bash deploy/install-operator-shortcuts.sh
```

This installs four intentionally short commands into `~/.local/bin`:

- `START` — start the systemd research daemon and return only after it is stably `active/running`.
- `STOP` — stop research immediately and return only after systemd reports `inactive/dead` with PID 0.
- `HUD` — open the live terminal operations HUD. It works while research is running or stopped.
- `STATUS` — show daemon state plus a one-shot plain-text HUD snapshot.

`STOP` is idempotent and is the preferred emergency/operator pause control. Stopping the daemon preserves durable PostgreSQL research state.

### Updating an existing editable deployment

Production normally uses an editable venv install pointed at the repository checkout. After a clean fast-forward, first verify that relationship instead of reinstalling the package:

```bash
"$HOME/adamArchives/Adam/.research-engine/venv/bin/python" - <<'PY'
import pathlib
import research_engine
print(pathlib.Path(research_engine.__file__).resolve())
PY
```

If the module resolves under `~/adamArchives/Adam/research-engine`, no pip reinstall is required; the fast-forwarded source is already active. If it does not, use a normal build-isolated editable install:

```bash
"$HOME/adamArchives/Adam/.research-engine/venv/bin/pip" install --no-deps -e "$HOME/adamArchives/Adam/research-engine"
```

Do not pass `--no-build-isolation` unless Hatchling is already installed in that venv.

## Operations HUD and allowance observations

The monitoring surfaces are read-only with respect to scientific state.

Record a ChatGPT/Codex plan reading from the product UI:

```bash
research-engine allowance-observe \
  --window 5h \
  --left-percent 86 \
  --resets-in "3h 20m" \
  --note "live usage UI observation"
```

Record weekly usage exactly as the UI presents it, for example `--window weekly --left-percent 96 --resets-in "5d 8h"`. `--used-percent` and absolute `--resets-at` remain available for programmatic callers.

Inspect a machine-readable snapshot:

```bash
research-engine ops-snapshot --run-id rf-open-world-corpus-expansion-v1
```

Launch the terminal HUD:

```bash
research-engine hud --run-id rf-open-world-corpus-expansion-v1
```

Use `--once --no-color` for a static/plain-text snapshot. The HUD reports
`CALIBRATING` rather than an exhaustion ETA until at least two usable
observations exist in the same allowance window.

PostgreSQL is authoritative for worker/job/usage/allowance telemetry. HUD
rendering failure must not change research state.

## Chat-visible ops mirror

The Chat-visible mirror is a bounded derivative of the same read-only PostgreSQL
snapshot. It deliberately whitelists operational fields and excludes run topic,
policy payloads, prompts, source text, credentials, environment values,
worker/host/PID identifiers, current job IDs, provider error detail, and raw
calibration summaries.

Render it locally without GitHub access:

```bash
research-engine ops-mirror \
  --run-id rf-open-world-corpus-expansion-v1 \
  --format markdown
```

Use `--format json` for the sanitized machine-readable form intended for a
future direct-tool/MCP surface.

GitHub publication is kept outside the research daemon:

```bash
bash deploy/publish-ops-mirror.sh current
bash deploy/publish-ops-mirror.sh rollup
```

`current` creates or edits one open issue titled
`[ops] research-engine live state`. `rollup` appends a durable issue comment,
providing low-frequency history without using Git commits as a telemetry sink.
The publisher requires an already-authenticated GitHub CLI (`gh auth status`)
and fails closed when credentials are unavailable. It does not start/stop the
daemon and does not mutate scientific PostgreSQL tables.

No publication timer is installed by this repository step. Validate GitHub CLI
availability/authentication on the Lambda first, then choose an update cadence.
Override `RESEARCH_ENGINE_GITHUB_REPO`,
`RESEARCH_ENGINE_OPS_ISSUE_TITLE`, or `RESEARCH_HUD_RUN_ID` when needed.

## VERIFY context economy

EXTRACT receives the bounded full-text PDF context needed to produce page-grounded
evidence. VERIFY does not resend the full paper. It receives the authoritative
evidence objects plus only their cited source pages.

`RESEARCH_ENGINE_VERIFY_PAGE_RADIUS=0` is the production default. Set it to
`1` only if calibration shows that adjacent-page context is needed for
verification quality. Missing evidence locators or unavailable cited pages fail
closed rather than silently broadening context.

This separation is intentional: VERIFY is a provenance check, not a second
full-paper extraction pass.

## VERIFY replay calibration

Before resuming sustained research, the optimized VERIFY context can be measured
against a completed historical VERIFY job without mutating scientific state.

Run with the daemon stopped:

```bash
bash deploy/run-verify-calibration.sh
```

The runner:

- requires `research-engine-daemon.service` to remain `inactive/dead`;
- runs a deterministic zero-model-call EXTRACT provenance audit before any
  semantic replay;
- records exactly one fresh pre-batch and one post-batch 5-hour/weekly usage-UI
  reading pair;
- records whole-percent LEFT readings as quantized/censored observations and
  records the displayed reset-duration precision;
- selects the lowest-cost and highest-cost historical VERIFY jobs by recorded
  input tokens, complementing the already-completed median sample;
- rehydrates each exact job using the current VERIFY page-window policy;
- performs exactly two STANDARD/Terra semantic replays;
- bypasses the semantic materializer, so no evidence object, verification event,
  checkpoint, gap, or human-review state is changed;
- persists each A/B measurement in `research_calibration_events`;
- exposes the results in the HUD CALIBRATION panel.

Direct use remains available:

```bash
research-engine calibrate-verify rf-open-world-corpus-expansion-v1 --selector lowest
research-engine calibrate-verify rf-open-world-corpus-expansion-v1 --selector median
research-engine calibrate-verify rf-open-world-corpus-expansion-v1 --selector highest
research-engine audit-extract-provenance rf-open-world-corpus-expansion-v1
```

The extraction audit invokes no semantic executor. It reports total PDF pages,
extraction-context pages/chars, evidence-bearing pages, evidence density, and
contiguous evidence-page clusters for each completed EXTRACT job.

A context optimization is not considered quality-safe merely because it saves
tokens. Compare both `verified_equal` and `accepted_evidence_ids_equal`.
If either differs, keep sustained research stopped and rerun only that source at
radius 1 before changing the global page-window policy.

Allowance reset timestamps derived from relative UI text are grouped with a
precision-aware tolerance. Raw observations remain durable, and a whole-percent
bucket transition is not treated as an exact percentage cost.

### Repeated radius A/B calibration

When a single replay cannot distinguish model nondeterminism from a context-window
effect, use the stopped-daemon repeated A/B runner on one explicit historical
VERIFY source:

```bash
bash deploy/run-verify-ab-batch.sh job:<source-id> 3
```

The second argument is replays per radius and defaults to 3. The runner caps it at
12 per radius, performs radius 0 first and radius 1 second, and never materializes
scientific state. It asks for exactly three whole usage-limit pastes: before the
radius-0 phase, between phases, and after the radius-1 phase. Each phase reports
agreement counts plus exact replay input/cached/output/reasoning-token statistics.
This separates exact token-cost comparison from the product UI's quantized plan-
allowance measurement.

The runner requires the daemon to remain inactive/dead before, during, and after
every semantic call and never starts sustained research.

## Allowance behavior

ChatGPT-authenticated Codex consumes the plan's Codex/Work allowance. Production
must pace or pause when that allowance is unavailable; it must never switch to a
paid API key automatically.

When Codex reports a plan/usage limit, the semantic provider enters a durable
PostgreSQL-backed backoff window. The affected job is requeued without consuming
its scientific retry budget, semantic workers keep heartbeating while they wait,
and deterministic work may continue. If Codex provides a retry/reset duration,
the daemon uses it; otherwise
`RESEARCH_ENGINE_CODEX_PLAN_LIMIT_RETRY_SECONDS` supplies the fallback.

Budget caps are enforced at lease admission. Per-tier saturation blocks only
that tier while other eligible tiers can continue; a run enters
`PAUSED_BUDGET` only when saturation leaves no eligible non-human work.

## Secrets

The production Codex path requires no `OPENAI_API_KEY`. Never commit ChatGPT,
Codex, API, or other credentials to this repository.


## Hybrid document routing follow-up

After the preserved Docling `standard_no_ocr` normalization replay has
completed, run the adaptive-routing, table-residual, and targeted-formula gates
together while the research daemon remains stopped:

```bash
bash deploy/run-document-hybrid-routing-followup.sh
```

The runner defaults to the preserved October 2 standard/no-OCR output root and
accepts another absolute output root as its first argument. It fails closed
unless the checkout is clean on `main`, the daemon is inactive/dead, the
normalized standard package exists, and exactly one existing normalized
Docling-native corpus can be located under the workspace-control temporary
artifact root.

Only the measured formula residual opens new inference. Formula target pages
come from the preserved standard Docling JSON. Each target page is processed
individually with OCR, table structure, code enrichment, picture analysis,
remote services, and general-document VLM disabled. The formula-specific
CodeFormulaV2 stage is the sole new semantic parser stage enabled by this gate.


## Visible temporary artifact directory

New operator-facing benchmark and diagnostic outputs must be written under:

```text
~/adamArchives/Adam/workspace-control-tmp/
```

rather than a hidden dot-directory. This keeps artifacts browseable and
downloadable from JupyterLab. Existing artifacts under
`~/adamArchives/Adam/.workspace-control/tmp/` remain valid legacy inputs and
must not be moved or regenerated merely to satisfy this convention.

The targeted TableFormer residual gate follows this rule:

```bash
bash deploy/run-tableformer-accurate-residual-gate.sh
```

It reads the preserved legacy standard/no-OCR package and table audit, selects
only table assets with genuinely empty FAST-mode grids, reuses the existing
Docling/TableFormer runtime and model cache, runs ACCURATE mode only on those
pages, and writes all new plan/raw/evaluation artifacts to the visible temporary
root. Set `WORKSPACE_CONTROL_TMP_ROOT` to another visible directory when
needed.


## Targeted Paddle table fallback

After TableFormer FAST and ACCURATE have both produced empty grids for the same
localized table assets, run:

```bash
bash deploy/run-paddle-table-fallback-gate.sh
```

The runner keeps the research daemon stopped, derives the same persistent
empty-grid targets from preserved evidence, creates an isolated PaddleOCR 3.7.0
/ PaddlePaddle 3.3.0 CUDA 12.6 runtime, and invokes PP-TableMagic only on
300-DPI crops of those table bboxes. It does not parse whole documents or
enable PP-StructureV3, formulas, charts, a document VLM, Marker, or MinerU.

The runtime and model cache live under `.research-engine`; all human-facing
benchmark outputs live under visible `workspace-control-tmp`.


## Shared local inference concurrency gate

After the formula and table quality routes are promoted, validate model
ownership and concurrency while the research daemon remains stopped:

```bash
bash deploy/run-local-inference-concurrency-gate.sh
```

The runner requires the existing CodeFormulaV2 and PP-TableMagic runtimes and
model caches. It does not create virtual environments or prefetch models.

On a dual-GPU host it probes both runtimes with real allocations, binds formula
and table ownership to distinct healthy GPUs, and runs a six-request mixed cold
wave followed by a two-request warm reuse wave. New human-facing reports and
per-request outputs are written beneath visible
`~/adamArchives/Adam/workspace-control-tmp/`.

The production daemon integration remains disabled by default until this gate
passes. Enabling `RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1` causes one broker
to be created per daemon process and shared across all deterministic workers.


## EXTRACT / VERIFY paper-package A/B gate

Run the bounded stopped-daemon downstream-context gate with:

```bash
bash deploy/run-extract-verify-paper-package-ab.sh
```

The runner discovers the preserved run containing exactly nine completed
EXTRACT and nine completed VERIFY jobs. It verifies every source artifact by
SHA-256, symlinks the immutable bytes into a visible workspace with a `.pdf`
suffix, and reuses the already-installed Docling native and standard/no-OCR
runtimes.

Each Docling runtime creates one converter and processes all nine PDFs in a
single process. The runner does not create environments, prefetch models, or
redownload weights.

A zero-model quote/provenance preflight must pass before semantic replay starts.
The representative semantic phase runs low/median/high EXTRACT and VERIFY
package arms and executes a current-PDF control only after package disagreement.
Scientific state is never materialized; operational calibration events and
human-facing JSON reports are written beneath
`~/adamArchives/Adam/workspace-control-tmp/`.

The final result distinguishes a package-context regression from model variance
and reports exact semantic token/latency deltas. It does not infer plan
allowance consumption from token counts.


## Lambda production acceptance

Before the bounded production-integration stage, synchronize the validated
shared local-inference settings into the live production env while the daemon is
stopped:

```bash
bash deploy/configure-production-local-inference.sh
```

The helper creates a timestamped backup beneath
`~/.research-engine/backups/` and prints the exact rollback command.

Then run:

```bash
bash deploy/run-lambda-production-acceptance.sh
```

The acceptance runner starts/checks PostgreSQL only. It does **not** start
`research-engine-daemon.service`. It requires a clean main checkout, the
installed systemd unit to match the repository unit, Codex-only production
configuration, one worker per tier for first acceptance, radius-0 VERIFY,
shared local inference enabled with bounded queues, all validated local
runtimes/caches, two healthy GPUs, a quiescent non-human queue, and the measured
localization/license contracts.

Its runtime smoke checks are bounded and read-only: small GPU allocations,
zero-spawn broker construction, one existing authoritative-PDF hydration, and
read-only operational reporting. Human-facing results are written beneath
visible `~/adamArchives/Adam/workspace-control-tmp/`.


## Bounded production integration

After Lambda production acceptance passes, synchronize the production env again
so the document-sidecar feature gate is present:

```bash
bash deploy/configure-production-local-inference.sh
```

Then run the finite integration gate:

```bash
bash deploy/run-bounded-production-integration.sh
```

The gate requires a quiescent non-human queue, fingerprints scientific tables,
builds one uncached preserved PDF sidecar and immediately proves a cache hit,
checks sidecar telemetry, starts the production daemon for a short idle
registration interval, verifies exactly one worker for each configured tier,
stops the daemon unconditionally, and requires the scientific fingerprint to be
unchanged.

The sidecar is a derived cache only. The gate does not submit a research run and
does not change EXTRACT/VERIFY away from authoritative-PDF semantic hydration.
