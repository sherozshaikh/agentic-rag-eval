# syntax=docker/dockerfile:1.7
# ==============================================================================
# agentic-rag-eval — multi-stage Dockerfile
# ------------------------------------------------------------------------------
# Stage 1 (builder): installs `uv`, resolves dependencies into a local .venv.
# Stage 2 (runtime): copies the venv + source, drops build tooling, runs as a
#                    non-root user.
#
# Design notes:
#   - No models baked in (embeddings/re-ranker/LLM are downloaded at runtime).
#   - Target image size ~500-600MB.
#   - Healthcheck hits FastAPI /health endpoint.
# ==============================================================================

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.5.11

# ------------------------------------------------------------------------------
# Stage 1 — builder
# ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Build-time system dependencies. `build-essential` is needed by some wheels
# (e.g. sentence-transformers deps); `curl` is used to fetch the uv installer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned) from the official installer.
ARG UV_VERSION
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv --version

WORKDIR /app

# Copy only the files needed to resolve dependencies first, so Docker can cache
# this layer across source changes.
COPY pyproject.toml README.md ./
# uv.lock is optional — copy it if present without failing the build.
COPY pyproject.tom[l] uv.loc[k] ./

# Resolve and install runtime dependencies into /app/.venv.
# We intentionally do NOT install the project itself here (no source yet);
# the source is copied in the runtime stage.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv \
    && uv sync --no-install-project --no-dev || uv pip install --python /app/.venv/bin/python -e .

# ------------------------------------------------------------------------------
# Stage 2 — runtime
# ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv \
    PYTHONPATH=/app/src \
    PORT=8000

# Runtime system deps — curl for the healthcheck only. Keep this list minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Create non-root user and app directories with correct ownership.
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid appuser --home /app --shell /bin/bash appuser \
    && mkdir -p /app /app/traces /app/mem0_storage /app/logs \
    && chown -R appuser:appuser /app

WORKDIR /app

# Copy the pre-built virtualenv from the builder stage.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application source. Only these directories are required at runtime.
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser prompts/ /app/prompts/
COPY --chown=appuser:appuser templates/ /app/templates/
COPY --chown=appuser:appuser static/ /app/static/
COPY --chown=appuser:appuser pyproject.toml README.md /app/

USER appuser

EXPOSE 8000

# Healthcheck — hits the FastAPI deep health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl --fail --silent --show-error "http://127.0.0.1:${PORT}/health" || exit 1

# tini as PID 1 for proper signal forwarding to uvicorn.
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "agentic_rag_eval.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
