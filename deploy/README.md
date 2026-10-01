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

## Explicit API-development profile only

`deploy/docker-compose.daemon.yml` is retained solely for explicit
API-development/testing. Its daemon service is behind the
`openai-api-development` Compose profile and sets
`RESEARCH_ENGINE_SEMANTIC_BACKEND=openai_api`.

Do not use that profile for production operation under the ChatGPT-plan
constraint.

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
