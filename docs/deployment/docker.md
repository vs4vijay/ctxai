# Docker Deployment

Build and run ctxai in containers.

## Build

```bash
docker build -t ctxai:1.0.0 .
```

The Dockerfile is a multi-stage build that produces a slim runtime image. It
installs the `[all]` extra plus FastAPI + uvicorn so the service layer works
out of the box.

## Run

```bash
docker run -d \
  --name ctxai \
  -p 8000:8000 \
  -e CTXAI_HOME=/data \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v ctxai-data:/data \
  ctxai:1.0.0
```

## Compose

`docker-compose.yml` starts ctxai together with Redis for distributed
rate-limiting / sessions:

```bash
docker compose up -d
docker compose logs -f ctxai
docker compose down
```

## Health checks

The image declares a HEALTHCHECK that hits `GET /api/v1/health`. Use
`docker inspect --format='{{.State.Health.Status}}' ctxai` to read it.

## Environment variables

| Name | Description | Default |
|------|-------------|---------|
| `CTXAI_HOME` | Persistent storage directory | `/data` |
| `CTXAI_ENV` | `dev` / `prod` (controls log format) | `prod` |
| `CTXAI_LOG_LEVEL` | One of DEBUG/INFO/WARNING/ERROR | `INFO` |
| `OPENROUTER_API_KEY` | API key for OpenRouter | unset |
| `ANTHROPIC_API_KEY` | API key for Anthropic | unset |
| `OPENAI_API_KEY` | API key for OpenAI | unset |

## Volumes

Persist `/data` (sessions DB, indexes, logs) across container restarts.
