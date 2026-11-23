"""AI가 쓸 수 있는 모든 말뭉치 레코드(근거 문단 또는 규칙)가 통과하는 인용 관문.

설계상 위치: 말뭉치 계약 계층(``corpus`` 패키지). 근거 색인(``clinical_evidence``)과 규칙 말뭉치 로더가
레코드를 받아들이기 전에 호출한다.

역할: 라이선스·원문 대조·폐기 여부·검수 상태로 인용 가능 여부를 판정한다. SQL 뷰 ``corpus.citable_evidence``가
데이터베이스에서 같은 규칙을 적용하며, 이 함수는 그 뒤에서 Python으로 한 번 더, 파일 기반 말뭉치에서는 단독으로 실행된다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

USABLE_VERIFICATIONS = frozenset({"primary", "local_db"})
EXCLUDED_CLAIM_STATUSES = frozenset({"폐기", "확인불가"})


def is_citable(record: Mapping[str, Any], *, include_unreviewed: bool, require_license: bool = False) -> bool:
    """레코드를 AI가 인용할 수 있는지 판정한다.

    역할: 레코드 ``ai_tier``가 ``A``이고, 출처가 하나 이상이며 모든 출처가 상업적 AI 이용 허가(``ai_use: A``)를 받았고,
        원문과 대조되었으며(``verification``이 ``primary``·``local_db``), 폐기·확인불가 주장이 아닐 때만 통과시킨다.
        의료진이 승인한(``approved``) 레코드는 항상, 초안(``draft``)은 ``include_unreviewed``일 때만 통과한다.
    입력:
        record: 말뭉치 레코드(근거 문단 또는 규칙).
        include_unreviewed: True이면 초안 레코드도 허용한다.
        require_license: True(출처 표기 전용 모드)이면 각 출처에 라이선스와 제목이 있어야 한다. 즉 화면에서 실제로
            출처를 밝힐 수 있어야 한다.
    출력: 인용 가능하면 True.
    """
    sources = record.get("sources")
    review = record.get("review_status")
    return (
        record.get("ai_tier") == "A"
        and isinstance(sources, list) and bool(sources)
        and all(isinstance(source, Mapping) and source.get("ai_use") == "A" for source in sources)
        and (not require_license or all(source.get("license") and source.get("title") for source in sources))
        and record.get("verification") in USABLE_VERIFICATIONS
        and record.get("claim_status") not in EXCLUDED_CLAIM_STATUSES
        and (review == "approved" or (include_unreviewed and review == "draft"))
    )
