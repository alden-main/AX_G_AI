"""한 요청이 말뭉치에서 쓰는 것: 같은 릴리스의 근거 검색과 규칙.

설계상 위치: 서비스 계층. ``runtime``이 파일 말뭉치(``StaticCorpus``) 또는 DB 말뭉치(``corpus_store.live.LiveCorpus``)를
고르고, API가 요청마다 ``CorpusSource.current``를 의존성으로 받아 근거 조립(``clinical_grounding``)에 넘긴다.

역할: 요청은 ``CorpusSource.current``를 한 번 읽고 끝까지 그 스냅샷을 쓰므로, 인용 문단·규칙 결과·DUR 검사가 두
릴리스에 섞이지 않는다. ``StaticCorpus``는 고정 스냅샷(파일 말뭉치, 테스트)을 들고, ``LiveCorpus``는 활성 말뭉치
릴리스나 DUR 데이터셋이 바뀌면 새 스냅샷으로 갈아 끼운다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ax_g_ai.services.clinical_evidence import EvidenceRetriever

if TYPE_CHECKING:
    from ax_g_ai.services.clinical_recommendation import ClinicalRecommender


@dataclass(frozen=True)
class CorpusSnapshot:
    """한 시점의 근거 검색기·규칙 추천기와 그 릴리스 식별자.

    역할: 요청 하나가 처음부터 끝까지 쓰는 불변 묶음. 없는 구성 요소는 ``None``이다.
    """

    evidence: EvidenceRetriever | None = None
    recommender: ClinicalRecommender | None = None
    # PostgreSQL의 말뭉치 릴리스(None: 파일 말뭉치). 준비된 환자 문맥이 이 값을 기록해 두어, 더 새 릴리스가
    # 활성화되면 규칙 사실을 다시 매핑한다.
    release_id: int | None = None
    dur_dataset_id: int | None = None


class CorpusSource(Protocol):
    """현재 말뭉치 스냅샷을 내주는 원천 인터페이스.

    역할: ``current``로 요청용 스냅샷을, ``database_reachable``로 준비 상태를 알려 주고, 종료 시 ``close``로 자원을 푼다.
    """

    @property
    def current(self) -> CorpusSnapshot: ...

    def database_reachable(self) -> bool | None:
        """말뭉치 DB가 지금 응답하는지(None: DB 없음). 동기 호출이며 예외를 내지 않는다. ``GET /ready``용."""
        ...

    def close(self) -> None: ...


class StaticCorpus:
    """바뀌지 않는 스냅샷 하나를 돌려주는 ``CorpusSource``.

    역할: 파일 말뭉치와 테스트용. DB가 없으므로 ``database_reachable``은 ``None``이다.
    """

    def __init__(self, snapshot: CorpusSnapshot | None = None) -> None:
        self._snapshot = snapshot or CorpusSnapshot()

    @property
    def current(self) -> CorpusSnapshot:
        return self._snapshot

    def database_reachable(self) -> bool | None:
        return None

    def close(self) -> None:
        return None
