# syntax=docker/dockerfile:1

# Keep the Python version aligned with pyproject.toml (requires-python >=3.12).
FROM ghcr.io/astral-sh/uv:0.9.16 AS uv

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8100

# uv is copied from its maintained image, avoiding a curl-based installer in
# the build stage.  The application itself runs without uv at runtime.
COPY --from=uv /uv /uvx /bin/

# Install the locked runtime environment before source code so dependency
# layers are retained when only application code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

# The ASGI server exposes /health, /docs, and the Bridge internal API routes.
ENTRYPOINT ["uvicorn", "ax_g_ai.api:app", "--host", "0.0.0.0", "--port", "8100"]
