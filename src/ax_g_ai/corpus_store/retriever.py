"""PostgreSQL 기반 하이브리드 근거 검색: 어휘 검색(프로세스 내 BM25) + pgvector.

설계상 위치: 말뭉치 저장 계층의 검색부. 스냅샷 빌더가 릴리스마다 검색기를 만들고, 임상 근거 서비스가 질문마다
``search``를 부른다. 연결 풀 생성(``open_pool``)과 활성 릴리스 조회(``active_release_id``)도 여기서 제공한다.

역할: ``PostgresEvidenceRetriever`` 하나가 말뭉치 릴리스 하나를 서빙한다. 인용 가능 문단(유일한 게이트인
``corpus.citable_evidence``)을 ``InMemoryEvidenceIndex``에 올려 어휘·개념 점수를 매기므로 어휘 검색 동작은 파일 기반
색인과 같다. 의미 후보는 pgvector에서 온다. 문단은 어휘 점수가 어휘 컷오프를 넘거나 **또는** 코사인 유사도가
``similarity_floor``를 넘으면 인용되고, 채택된 문단은 상호 순위 융합(RRF)으로 정렬된다. 질의 시점에 임베딩 서버나
DB가 실패해도 메모리에 올린 어휘 색인이 계속 답한다.

더 새 릴리스는 ``corpus_store.live.LiveCorpus``가 감지해, 같은 릴리스의 규칙과 함께 새 검색기를 만들고 둘을 한 번에
교체한다.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ax_g_ai.corpus_store.embedding import EmbeddingError, TeiEmbeddingClient, vector_literal
from ax_g_ai.corpus_store.rules import read_rows
from ax_g_ai.services.clinical_evidence import EvidenceChunk, InMemoryEvidenceIndex, lexical_cutoff, prioritize

logger = logging.getLogger("uvicorn.error")
_RRF_K = 60


class CorpusUnavailableError(RuntimeError):
    """DB에 활성 말뭉치 릴리스가 없을 때의 오류.

    역할: 호출자가 DB 연결 오류와 구분해 파일 기반 말뭉치로 대체하거나 시작을 실패시키게 한다.
    """


def open_pool(database_url: str, *, max_size: int = 10, wait_seconds: float = 0) -> ConnectionPool:
    """요청 경로 쿼리용 연결 풀을 연다.

    역할: autocommit·3초 연결 제한·``dict_row`` 연결로 풀을 만들고 연결될 때까지 기다린다. 실패하면 풀의 재연결
    작업자를 멈추고 예외를 다시 던진다.
    입력:
        database_url: PostgreSQL 연결 URL.
        max_size: 풀의 최대 연결 수.
        wait_seconds: API와 함께 시작하는 DB를 기다리는 시간. 풀은 이 시간(최소 3초) 동안 백오프하며 재연결을 시도한다.
    출력: 열린 ``ConnectionPool``.
    예외: DB가 내려가 있으면 ``psycopg.Error``(``PoolTimeout``).
    """
    pool = ConnectionPool(
        database_url, min_size=1, max_size=max_size, timeout=3, open=False,
        kwargs={"autocommit": True, "connect_timeout": 3, "row_factory": dict_row},
    )
    try:
        pool.open(wait=True, timeout=max(3.0, wait_seconds))
    except BaseException:
        pool.close()  # 호출자가 파일로 대체하기 전에 풀의 재연결 작업자를 멈춘다
        raise
    return pool


def active_release_id(conn: psycopg.Connection) -> int:
    """활성 말뭉치 릴리스 ID. 없으면 ``CorpusUnavailableError``."""
    row = conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchone()
    if row is None:
        raise CorpusUnavailableError("no active corpus release")
    return _first(row)


def load_index(
    conn: psycopg.Connection, release_id: int, *, include_unreviewed: bool, require_license: bool = False,
) -> InMemoryEvidenceIndex:
    """``release_id``의 인용 가능 문단으로 메모리 색인을 만든다(게이트 뷰는 활성 릴리스만 읽으므로 활성 릴리스여야 한다).

    역할: 단독 검색기(``open``)용. 서비스는 ``read_rows``를 한 번 읽어 색인과 규칙을 함께 만든다.
    입력:
        conn: PostgreSQL 연결.
        release_id: 활성 릴리스 ID.
        include_unreviewed: 검수되지 않은 문단도 포함할지.
        require_license: True면 출처마다 라이선스·제목이 있는 문단만 인용한다(출처 표기 전용 모드).
    출력: ``InMemoryEvidenceIndex``.
    """
    # SQL 뷰 뒤의 두 번째 게이트로 Python에서 is_citable()을 다시 적용한다.
    return InMemoryEvidenceIndex.from_rows(read_rows(conn, release_id), include_unreviewed=include_unreviewed, require_license=require_license)


class PostgresEvidenceRetriever:
    """메모리에 올린 릴리스 하나에 대한 근거 검색기.

    역할: 어휘 점수(메모리 색인)와 pgvector 의미 후보를 합쳐 인용할 문단을 고르고 RRF로 정렬한다.
    연결 풀과 임베딩 클라이언트는 공유받은 것이며 소유하지 않는다(``open``으로 만든 경우만 풀을 닫는다).
    """

    def __init__(
        self, pool: ConnectionPool, release_id: int, index: InMemoryEvidenceIndex, *, embedder: TeiEmbeddingClient | None,
        model_id: str, include_unreviewed: bool, similarity_floor: float = 0.6, candidates: int = 30,
    ) -> None:
        """검색기 설정을 보관한다.

        역할: 풀·릴리스·색인·임베딩 클라이언트와 검색 매개변수를 저장한다.
        입력:
            pool: 의미 검색 쿼리에 쓸 공유 연결 풀.
            release_id: 이 검색기가 서빙하는 릴리스 ID.
            index: 그 릴리스의 인용 가능 문단 색인.
            embedder: 질문 임베딩 클라이언트. ``None``이면 어휘 검색만 한다.
            model_id: 저장된 임베딩을 고를 모델 ID.
            include_unreviewed: False면 의미 후보를 ``review_status = 'approved'``로 제한한다.
            similarity_floor: 의미 후보만으로 채택할 최소 코사인 유사도.
            candidates: 어휘·의미 각각 RRF에 넣을 후보 수(의미 검색 ``LIMIT``).
        출력: 없음.
        """
        self._pool = pool
        self._release_id = release_id
        self._index = index
        self._embedder = embedder
        self._model_id = model_id
        self._include_unreviewed = include_unreviewed
        self._similarity_floor = similarity_floor
        self._candidates = candidates

    @classmethod
    def open(
        cls, database_url: str, *, embedder: TeiEmbeddingClient | None, model_id: str, include_unreviewed: bool,
        require_license: bool = False, **options: Any,
    ) -> PostgresEvidenceRetriever:
        """자체 풀을 가진 활성 릴리스 검색기를 만든다(도구·테스트용, 서비스는 ``LiveCorpus``를 쓴다).

        역할: 풀을 열고 활성 릴리스의 색인을 읽는다. 실패하면 풀을 닫고 예외를 다시 던진다.
        입력:
            database_url: PostgreSQL 연결 URL.
            embedder: 질문 임베딩 클라이언트 또는 ``None``.
            model_id: 임베딩 모델 ID.
            include_unreviewed: 검수되지 않은 문단 포함 여부.
            require_license: True면 출처마다 라이선스·제목이 있는 문단만 인용한다.
            options: 생성자에 넘길 나머지 설정(``similarity_floor``·``candidates``).
        출력: ``PostgresEvidenceRetriever``.
        예외: DB 오류(``psycopg.Error``) 또는 활성 릴리스 없음(``CorpusUnavailableError``).
        """
        pool = open_pool(database_url)
        try:
            with pool.connection() as conn:
                release_id = active_release_id(conn)
                index = load_index(conn, release_id, include_unreviewed=include_unreviewed, require_license=require_license)
        except BaseException:
            pool.close()
            raise
        return cls(pool, release_id, index, embedder=embedder, model_id=model_id, include_unreviewed=include_unreviewed, **options)

    def close(self) -> None:
        """풀을 닫는다(``open``으로 만든 검색기에만 쓴다)."""
        self._pool.close()

    def __len__(self) -> int:
        return len(self._index)

    @property
    def release_id(self) -> int:
        """서빙 중인 릴리스 ID."""
        return self._release_id

    @property
    def unreviewed_count(self) -> int:
        """색인에 포함된 미검수 문단 수."""
        return self._index.unreviewed_count

    def semantic(self, question: str) -> list[tuple[str, float]]:
        """질문과 의미가 가까운 문단을 pgvector에서 찾는다.

        역할: 질문을 임베딩해 이 릴리스의 인용 가능 문단 임베딩과 코사인 거리로 비교한다(미검수 포함 설정 반영).
        임베딩 서버·DB 오류는 경고 로그만 남기고 빈 결과로 처리한다.
        입력:
            question: 사용자 질문.
        출력: (chunk_id, 코사인 유사도) 목록, 유사도 높은 순. 임베딩을 쓸 수 없으면 빈 목록.
        """
        if self._embedder is None:
            return []
        try:
            vector = vector_literal(self._embedder.embed([question])[0])
            with self._pool.connection() as conn:
                rows = conn.cursor(row_factory=dict_row).execute(
                    "SELECT c.chunk_id, 1 - (e.embedding <=> %(v)s::vector) AS similarity "
                    "FROM corpus.citable_evidence c JOIN corpus.evidence_embedding_1024 e "
                    "  ON e.model_id = %(model)s AND e.body_sha256 = c.body_sha256 "
                    "WHERE c.release_id = %(release)s AND (%(unreviewed)s OR c.review_status = 'approved') "
                    "ORDER BY e.embedding <=> %(v)s::vector LIMIT %(k)s",
                    {"v": vector, "model": self._model_id, "release": self._release_id,
                     "unreviewed": self._include_unreviewed, "k": self._candidates},
                ).fetchall()
        except (EmbeddingError, psycopg.Error) as error:
            logger.warning("clinical_evidence_semantic_failed error_type=%s", type(error).__name__)
            return []
        return [(row["chunk_id"], float(row["similarity"])) for row in rows]

    def search(self, question: str, limit: int) -> tuple[EvidenceChunk, ...]:
        """어휘·의미 검색을 합쳐 인용할 근거 문단을 고른다.

        역할: 어휘 컷오프를 넘은 문단과 ``similarity_floor``를 넘은 의미 후보를 채택하고, 양쪽 상위 ``candidates``개의
        RRF 점수(``1 / (60 + 순위)`` 합) 순으로 정렬한 뒤 ``prioritize``로 대체된(참고용) 문단을 뒤로 보내고 ``limit``개로 자른다.
        입력:
            question: 사용자 질문.
            limit: 돌려줄 최대 문단 수.
        출력: ``EvidenceChunk`` 튜플. 질문이 비었거나 ``limit``이 0 이하면 빈 튜플.
        """
        if not question.strip() or limit <= 0:
            return ()
        index = self._index
        lexical = index.scored(question)
        cutoff = lexical_cutoff(lexical)
        semantic = self.semantic(question)
        accepted = {chunk.chunk_id for score, chunk in lexical if score >= cutoff}
        accepted |= {chunk_id for chunk_id, similarity in semantic if similarity >= self._similarity_floor}
        fused: dict[str, float] = defaultdict(float)
        for rank, (_score, chunk) in enumerate(lexical[: self._candidates], start=1):
            fused[chunk.chunk_id] += 1 / (_RRF_K + rank)
        for rank, (chunk_id, _similarity) in enumerate(semantic, start=1):
            fused[chunk_id] += 1 / (_RRF_K + rank)
        ordered = sorted((chunk_id for chunk_id in accepted if index.chunk(chunk_id) is not None),
                         key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        return tuple(prioritize(index.chunk(chunk_id) for chunk_id in ordered)[:limit])  # type: ignore[misc]


def _first(row: Any) -> Any:
    """튜플·dict 행 팩토리 어느 쪽이든 행의 첫 번째 열 값."""
    return row[0] if isinstance(row, tuple) else next(iter(row.values()))
