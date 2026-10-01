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
