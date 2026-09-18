"""D-06 승인 지식문서의 조회와 인용 정보 보존."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


@dataclass(frozen=True)
class KnowledgeDocument:
    """의료진 챗봇이 근거로 표시할 수 있는 버전 관리 지식문서.

    ``document_id``, ``title``, ``version``, ``citation_location``은 화면 응답의 근거를
    구성한다. ``approved``가 거짓이거나 유효기간 밖인 문서는 어떤 질의에도 반환하지
    않는다.
    """

    document_id: str
    title: str
    version: str
    published_on: date
    citation_location: str
    approved: bool
    valid_from: date
    valid_to: date | None


class ApprovedKnowledgeStore:
    """D-06의 승인 상태·유효기간을 결정적으로 검사하는 지식 저장소 경계.

    현재 구현은 주입받은 문서 집합만 조회하는 메모리 구현이다. 향후 검색 저장소를
    연결하더라도 이 클래스가 승인·버전·인용 위치 검사 뒤의 문서만 반환해야 한다.
    """

    def __init__(self, documents: Iterable[KnowledgeDocument]) -> None:
        self._documents = {document.document_id: document for document in documents}

    def get_approved(self, document_ids: Iterable[str], on_date: date) -> tuple[KnowledgeDocument, ...]:
        """지정된 식별자 중 승인되고 유효한 문서만 입력 순서대로 반환한다.

        Args:
            document_ids: 질의에 근거로 연결하려는 문서 식별자 iterable이다.
            on_date: 문서 유효기간을 판정하는 기준일이다.

        Returns:
            승인 여부, 유효기간, 인용 위치가 모두 충족된 ``KnowledgeDocument`` 튜플이다.
            미승인·만료·없는 문서는 결과에 넣지 않는다.
        """
        approved: list[KnowledgeDocument] = []
        for document_id in document_ids:
            document = self._documents.get(document_id)
            if (
                document
                and document.approved
                and document.valid_from <= on_date
                and (document.valid_to is None or on_date <= document.valid_to)
                and document.citation_location.strip()
            ):
                approved.append(document)
        return tuple(approved)
