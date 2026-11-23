"""자체 호스팅 Text Embeddings Inference(TEI) 서버용 임베딩 클라이언트.

설계상 위치: 말뭉치 저장 계층의 외부 연동부. 적재기(``loader``)가 근거 문단을, 검색기(``retriever``)가 질문을
임베딩할 때 쓴다.

역할: TEI ``/embed`` API를 배치로 호출하고 응답의 형태·차원·값을 검증한다. 기본 모델은 ``nlpai-lab/KURE-v1``
(MIT, 한국어로 미세조정한 bge-m3, 1024차원, CLS 풀링)이다. TEI는 내부 compose 네트워크 안에서 돌기 때문에
질문 텍스트가 배포 환경 밖으로 나가지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

EMBEDDING_DIMENSIONS = 1024
_BATCH = 16


class EmbeddingError(RuntimeError):
    """임베딩 서버에 연결할 수 없거나 예상과 다른 응답을 받았을 때의 오류.

    역할: 호출자가 HTTP·JSON 오류를 하나의 예외로 받아 어휘 검색만으로 계속하거나 경고로 남기게 한다.
    """


class TeiEmbeddingClient:
    """TEI 서버 하나에 대한 동기 임베딩 클라이언트.

    역할: 프로세스당 하나의 ``httpx.Client``로 연결을 재사용하며 텍스트 목록을 단위 정규화된 벡터로 바꾼다.
    """

    def __init__(
        self, base_url: str, *, timeout_seconds: float = 10.0, dimensions: int = EMBEDDING_DIMENSIONS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._dimensions = dimensions
        # 프로세스당 클라이언트 하나: keep-alive 연결을 질문 사이에 재사용한다(클라이언트는 스레드 안전).
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_seconds, transport=transport)

    @property
    def dimensions(self) -> int:
        """기대하는 임베딩 차원 수."""
        return self._dimensions

    def close(self) -> None:
        """HTTP 연결을 닫는다."""
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """텍스트마다 단위 정규화된 임베딩 하나를 돌려준다(너무 긴 입력은 TEI가 잘라낸다).

        역할: 16개씩 묶어 ``/embed``에 ``normalize``·``truncate``를 켜고 요청한 뒤, 배치마다 응답을 검증해 이어 붙인다.
        입력:
            texts: 임베딩할 텍스트 목록.
        출력: 입력 순서와 같은 벡터 목록(각 벡터는 ``float`` 목록).
        예외: HTTP 오류·잘못된 JSON·형태가 다른 응답이면 ``EmbeddingError``.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = list(texts[start:start + _BATCH])
            try:
                response = self._client.post("/embed", json={"inputs": batch, "normalize": True, "truncate": True})
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as error:
                raise EmbeddingError(f"embedding request failed: {type(error).__name__}") from error
            vectors.extend(self._validated(payload, len(batch)))
        return vectors

    def _validated(self, payload: object, expected: int) -> list[list[float]]:
        """TEI 응답이 기대한 개수·차원의 유한한 숫자 벡터인지 확인한다.

        역할: 응답 형태를 검사하고 값을 ``float``로 바꾼다.
        입력:
            payload: ``/embed`` 응답 JSON.
            expected: 이 배치에서 보낸 텍스트 수.
        출력: 검증된 벡터 목록.
        예외: 목록이 아니거나 개수·차원이 다르거나 유한하지 않은 값이 있으면 ``EmbeddingError``.
        """
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
    """벡터를 pgvector 텍스트 표기(``[x,y,...]``)로 바꾼다. 별도 Python 어댑터 없이 SQL에 넘길 수 있다."""
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"
