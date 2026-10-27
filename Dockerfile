# syntax=docker/dockerfile:1

# Stage 1: Build stage with uv
FROM python:3.12-slim AS builder

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency files for cached dependency installation
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and packages
COPY packages/ packages/
COPY apps/ apps/
COPY eval/ eval/

# Install the project into the virtual environment
RUN uv sync --frozen --no-dev

# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runner

# Install uv for runtime CLI commands
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/packages/memory_engine/src:/app"

WORKDIR /app

# Copy virtual environment and application code from builder
COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml ./
COPY packages/ packages/
COPY apps/ apps/
COPY eval/ eval/
COPY ops/ ops/

# Copy entrypoint script
COPY ops/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
