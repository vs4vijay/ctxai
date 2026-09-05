# Multi-stage Dockerfile for ctxai.
# Builds a slim runtime image (~150MB) suitable for the service layer.

FROM python:3.13-slim AS builder
WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install --prefix=/install ".[all]" && \
    pip install --prefix=/install fastapi "uvicorn[standard]"


FROM python:3.13-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CTXAI_HOME=/data \
    CTXAI_ENV=prod

COPY --from=builder /install /usr/local
COPY src/ /app/src/

RUN useradd --create-home --uid 1000 ctxai && \
    mkdir -p /data && chown -R ctxai:ctxai /data /app

USER ctxai
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=3).status == 200 else 1)" \
    || exit 1

CMD ["python", "-m", "uvicorn", "ctxai.service.api_server:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
