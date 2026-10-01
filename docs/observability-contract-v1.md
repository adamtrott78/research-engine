# Research Engine Observability Contract v1

Status: implementation contract
Baseline: `6a423e0fd38283687cae2eb060405a2a82643450`

## Purpose

Provide a low-overhead operations layer for the autonomous research engine before sustained semantic execution resumes. PostgreSQL remains authoritative. The HUD, reports, GitHub mirrors, and later MCP/Slack surfaces are derived views and must not mutate scientific state.

## Authority and failure isolation

Authoritative operational records remain in PostgreSQL:
- `research_runs`
- `research_workers`
- `research_jobs`
- `research_job_attempts`
- `research_usage_events`
- `research_provider_backoffs`
- scientific pipeline tables already owned by the research workflow.

Observability failures must not fail a research run. A HUD/rendering/publisher failure is never grounds to mutate evidence, checkpoints, gaps, novelty decisions, or human-review state.

## Agent identity

Two identities are required.

### Logical agent identity

Stable across daemon restarts and used for long-horizon tuning:

- `deterministic:0`
- `semantic:cheap:0`
- `semantic:standard:0`
- `semantic:strong:0`

Future concurrency increments use the same `<role>:<tier>:<index>` convention.

### Worker session identity

`research_workers.id` remains the ephemeral process/session identity and continues to include daemon host/PID information. Every worker row gains explicit operational metadata:

- `logical_agent_id`
- `host_name`
- `process_id`
- `code_sha`
- `backend`
- `model`
- `reasoning_effort`

The existing `created_at` is the session start and `last_heartbeat_at` is liveness. Historical worker rows are retained.

## Usage event contract

A successful semantic job produces one immutable `research_usage_events` row. It is attributed to:

- run
- job
- job type
- executor tier
- logical agent
- worker session
- final attempt number
- provider
- model
- reasoning effort

Usage fields:

- input tokens
- cached input tokens when supplied by Codex
- output tokens
- reasoning output tokens when supplied by Codex
- semantic call count (=1 per row)
- execution wall time
- queue wait time
- native web-search event count

Legacy rows may have zero/NULL values for newly introduced fields. No historical token values are invented.

## Allowance observations

Plan allowance is not inferred as an exact function of tokens.

New table: `research_allowance_observations`.

Each row records:
- observation timestamp
- window: `FIVE_HOUR` or `WEEKLY`
- percent used [0,100]
- optional reset timestamp
- source (initially `MANUAL_UI`)
- optional note
- current maximum usage-event ID as a durable correlation watermark

The first seed observation is the user's approximate 16% reading for the current five-hour window during the 2026-10-01 acceptance/launch work. Because the exact observation time and reset time are not known, it is recorded as approximate and is not sufficient by itself to produce a burn-rate ETA.

## Forecast semantics

Forecasts are derived and may be absent.

For each window:
1. Find the latest two or more observations that belong to the same reset window when reset identity is known; otherwise use only monotonic observations without an intervening drop.
2. Burn rate = delta percent used / elapsed wall-clock hours.
3. Exhaustion ETA = (100 - latest percent used) / positive burn rate.
4. If a reset timestamp is known, report both exhaustion ETA and reset time.
5. Never extrapolate from a single observation.
6. Never claim exact token-to-plan conversion.

Confidence:
- `CALIBRATING`: fewer than 2 usable observations.
- `LOW`: 2 observations or less than 30 minutes of observed interval.
- `MEDIUM`: >=3 usable observations spanning >=30 minutes.
- `HIGH`: >=5 usable observations spanning >=2 hours with stable burn-rate coefficient of variation <=25%.

Per-agent plan consumption starts as an explicitly labeled token/call proxy. Direct allowance-share attribution is withheld until enough observation intervals exist to fit and validate model/tier weights.

## Derived metric formulas

All rolling windows are computed from authoritative timestamps at query time.

### Queue and latency
- queue depth = QUEUED jobs.
- oldest queue age = now - minimum available_at among QUEUED jobs.
- attempt runtime = now - started_at for RUNNING attempts, otherwise finished_at - started_at.
- completed job latency = final attempt finished_at - job created_at.
- queue wait = attempt started_at - job available_at, lower-bounded at zero.

### Pipeline funnel
For the selected run:
- discovered = candidate count in completed DISCOVERY result, bounded by its candidate_limit when present.
- retained = retained IDs in SCREEN result that map to bounded discovery candidates.
- acquired = COMPLETED ACQUIRE jobs.
- extracted = COMPLETED EXTRACT jobs.
- verified = COMPLETED VERIFY jobs.
- checkpointed = COMPLETED CHECKPOINT jobs.
- syntheses = COMPLETED SYNTHESIZE jobs.
- gaps generated = persisted gap candidates scoped to the run.
- novelty searches = completed/cancelled/running NOVELTY_SEARCH jobs.
- novelty arbitrations = completed NOVELTY_ARBITRATE jobs / novelty challenge rows.
- human reviews = pending/approved/rejected requests.

### Throughput
For configurable rolling windows (default 1h and 5h):
- jobs/hour by job type
- acquired sources/hour
- extracted sources/hour
- verified sources/hour
- evidence objects/hour
- gaps/hour
- novelty searches/hour
- semantic calls/hour
- input/output tokens/hour

### Efficiency
When denominators are nonzero:
- input tokens per verified source
- total tokens per verified source
- semantic calls per verified source
- total tokens per evidence object
- total tokens per challenged gap
- wall-clock minutes per verified source

These are descriptive measurements, not scientific quality scores.

## HUD contract

Command: `research-engine hud`.

Default refresh: 2 seconds. Read-only. ANSI terminal rendering; no curses dependency required.

Panels:

### Header
- daemon state: supplied from an optional systemd probe when available, otherwise DB-only
- selected run ID/status/phase/human gate
- captured time
- provider backoff state

### Allowance
- FIVE_HOUR latest used %
- FIVE_HOUR burn %/h, ETA, reset, confidence
- WEEKLY same fields
- explicit `CALIBRATING` when insufficient observations

### Agents
One row per logical agent/session:
- logical agent
- session state
- model/tier
- current job type + abbreviated job ID
- job elapsed
- heartbeat age
- completed jobs
- failures/retries/deferred-plan-limit count
- input/output tokens
- share of semantic input/output tokens

### Pipeline
- discovered -> retained -> acquired -> extracted -> verified
- evidence objects
- checkpoint/synthesis count
- gaps -> novelty searches -> arbitrations -> human review

### Throughput
- 1h and 5h sources/hour
- evidence/hour
- semantic calls/hour
- input/output tokens/hour
- tokens per verified source

### Queue / Alerts
- queued/running/waiting-human counts
- oldest queue age
- failed jobs
- expired leases
- active provider backoffs
- oldest pending human review

## CLI surfaces

### `research-engine ops-snapshot`
Read-only JSON document containing the exact data used by the HUD. Supports `--run-id` and rolling-window options.

### `research-engine hud`
Repeatedly renders `ops-snapshot`. Supports:
- `--run-id`
- `--refresh-seconds`
- `--once`
- `--no-color`

### `research-engine allowance-observe`
Append-only operational input:
- `--window 5h|weekly`
- `--used-percent`
- optional `--resets-at`
- optional `--note`
- optional `--approximate`

This command writes only the allowance-observation table.

## Retention

v1 retains all job attempts, usage events, worker sessions, allowance observations, and provider backoffs. No automated deletion or aggregation replaces raw records.

Later rollups may improve query speed but remain derived.

## Chat-visible mirror boundary

The later GitHub/MCP/Slack layer consumes `ops-snapshot`; it must not invent a parallel metric implementation.

High-frequency truth stays in PostgreSQL. A GitHub live-ops issue is a low-frequency derived snapshot/audit surface.

## Chat reasoning bridge boundary

Later reasoning requests contain bounded context and durable references to run/job/checkpoint records. Chat responses are advisory.

A Chat response cannot directly:
- mark evidence MACHINE_CHECKED,
- alter a knowledge checkpoint,
- accept a novelty claim,
- approve a human review,
- or bypass local materialization/validation.

Preferred future direct read/response tool: private research-engine MCP plugin.
Optional wake-up/event bus: Slack.
Durable audit/history: GitHub.
