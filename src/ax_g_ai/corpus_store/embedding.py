"""Client for a self-hosted Text Embeddings Inference (TEI) server.

The default model is ``nlpai-lab/KURE-v1`` (MIT, bge-m3 fine-tuned on Korean, 1024
dimensions, CLS pooling). TEI runs inside the internal compose network, so question
text never leaves the deployment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

EMBEDDING_DIMENSIONS = 1024
_BATCH = 16


class EmbeddingError(RuntimeError):
    """The embedding server is unreachable or returned an unexpected payload."""


class TeiEmbeddingClient:
    def __init__(
        self, base_url: str, *, timeout_seconds: float = 10.0, dimensions: int = EMBEDDING_DIMENSIONS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._dimensions = dimensions
        self._transport = transport

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Unit-normalized embeddings, one per text (TEI truncates over-long input)."""
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            for start in range(0, len(texts), _BATCH):
                batch = list(texts[start:start + _BATCH])
                try:
                    response = client.post(f"{self._base_url}/embed", json={"inputs": batch, "normalize": True, "truncate": True})
                    response.raise_for_status()
                    payload = response.json()
                except (httpx.HTTPError, ValueError) as error:
                    raise EmbeddingError(f"embedding request failed: {type(error).__name__}") from error
                vectors.extend(self._validated(payload, len(batch)))
        return vectors

    def _validated(self, payload: object, expected: int) -> list[list[float]]:
        if not isinstance(payload, list) or len(payload) != expected:
            raise EmbeddingError("embedding response has the wrong shape")
        result = []
        for vector in payload:
            if (
                not isinstance(vector, list) or len(vector) != self._dimensions
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)
            ):
                raise EmbeddingError("embedding vector has the wrong dimension or values")
            result.append([float(x) for x in vector])
        return result


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector text form, so no extra Python adapter is needed."""
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"
