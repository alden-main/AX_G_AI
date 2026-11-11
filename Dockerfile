# syntax=docker/dockerfile:1

# Keep the Python version aligned with pyproject.toml (requires-python >=3.12).
FROM ghcr.io/astral-sh/uv:0.9.16 AS uv

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    AX_G_REFERENCE_DATA_DB_PATH=/app/data/reference_data.sqlite \
    AX_G_DUR_SNAPSHOT_DIR=/app/data/dur \
    AX_G_EVIDENCE_CORPUS_DIR=/app/text_data/corpus \
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
COPY data/reference_data.sqlite ./data/reference_data.sqlite
RUN test -s data/reference_data.sqlite
# 심평원 DUR 의약품 목록 스냅샷(ax-g-ai-build-dur 로 생성). dur-loader 서비스가 PostgreSQL dur 스키마에 적재하고,
# API는 활성 데이터셋을 DB에서 읽는다(처방 병용·연령·임부금기, 노인주의 점검).
COPY data/dur ./data/dur
RUN python -c "import sys, json; m = json.load(open('data/dur/manifest.json')); sys.exit(0 if m['tables']['product']['count'] > 0 else 1)"
# Generated corpus (build_corpus.py output). The corpus-loader service loads it into PostgreSQL;
# without a database the API searches these files directly.
COPY text_data/corpus ./text_data/corpus
RUN test -s text_data/corpus/chunks.jsonl
RUN uv sync --locked --no-dev
# Precomputed passage embeddings (text_data/tools/build_embeddings.py) must cover the corpus,
# otherwise the first start embeds on CPU for minutes. Override only on purpose:
#   docker build --build-arg ALLOW_STALE_EMBEDDINGS=1 .
ARG ALLOW_STALE_EMBEDDINGS=0
RUN python -m ax_g_ai.corpus_store.precomputed --check text_data/corpus || [ "$ALLOW_STALE_EMBEDDINGS" = "1" ]

# The ASGI server exposes /health, /docs, and the Bridge internal API routes.
ENTRYPOINT ["uvicorn", "ax_g_ai.api:app", "--host", "0.0.0.0", "--port", "8100"]
