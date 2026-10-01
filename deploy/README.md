# 24/7 deployment

The daemon is a replaceable supervisor over PostgreSQL-backed durable research
state. Restarting or replacing the process does not replace scientific authority.

## Preferred: Docker Compose

1. Create a private environment file outside the repository, for example:
   `~/adamArchives/Adam/.research-engine/research-engine.env`.
2. Copy values from `deploy/research-engine.env.example` and set
   `OPENAI_API_KEY` locally.
3. From the repository root run:

```bash
docker compose --env-file "$HOME/adamArchives/Adam/.research-engine/research-engine.env" \
  -f docker-compose.yml -f deploy/docker-compose.daemon.yml up -d --build
```

The Compose deployment provides PostgreSQL, the daemon, automatic restart, and a
persistent content-addressed artifact volume.

Useful commands:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.daemon.yml ps
docker compose -f docker-compose.yml -f deploy/docker-compose.daemon.yml logs -f daemon
docker compose -f docker-compose.yml -f deploy/docker-compose.daemon.yml exec daemon research-engine overnight-report --hours 12
```

## Native user systemd

The systemd template assumes the repository is at
`~/adamArchives/Adam/research-engine` and `research-engine` is installed in a
PATH visible to the user service.

Store the private environment at:

`~/adamArchives/Adam/.research-engine/research-engine.env`

Install the template as:

`~/.config/systemd/user/research-engine-daemon.service`

Then reload and enable it with the normal user-level systemd commands.

The service uses `Restart=always`. It must not use `/tmp` for human-transfer
or durable artifact storage.

## Secrets

Never commit `OPENAI_API_KEY` or other credentials. The checked-in env file is
only a key/value template with an empty secret field.
