FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[openai]" \
    && useradd --system --uid 10001 --create-home research-engine \
    && mkdir -p /var/lib/research-engine/artifacts \
    && chown -R research-engine:research-engine /var/lib/research-engine /app

USER research-engine

ENTRYPOINT ["research-engine"]
CMD ["daemon"]
