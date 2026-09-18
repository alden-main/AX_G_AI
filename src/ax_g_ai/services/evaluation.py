"""D-10 의료진 챗봇의 임상 평가 활성화 게이트.

이 모듈은 모델 품질을 자동 채점하지 않는다. 임상기관이 승인한 평가셋과 평가 결과를
버전별로 기록하고, 그 승인 기록이 없는 후보는 운영 챗봇 호출을 허용하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


class EvaluationNotApprovedError(PermissionError):
    """후보 버전에 승인된 임상 평가 기록이 없어 챗봇을 활성화할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class ChatVersion:
    """평가와 운영 활성화를 연결하는 Provider·모델·프롬프트·지식 버전 조합."""

    provider: str
    model_version: str
    prompt_version: str
    knowledge_version: str


@dataclass(frozen=True)
class EvaluationRecord:
    """임상기관이 검토한 특정 챗봇 후보의 평가·활성화 승인 기록.

    Attributes:
        version: 평가한 ``ChatVersion``이며 운영 후보와 완전히 일치해야 한다.
        evaluation_set_id: 비식별·사용 승인이 확인된 평가셋 식별자다.
        evaluated_on: 평가 결과를 확정한 기준일이다.
        evaluator_id: 결과에 책임을 지는 승인된 평가자 식별자다.
        approved_for_activation: 임상 책임자가 활성화를 승인했는지 나타낸다.
        approval_reference: 승인 문서 또는 변경 기록의 추적 식별자다.
    """

    version: ChatVersion
    evaluation_set_id: str
    evaluated_on: date
    evaluator_id: str
    approved_for_activation: bool
    approval_reference: str


class EvaluationGate:
    """승인된 평가 기록이 있는 버전만 의료진 챗봇 호출을 허용한다.

    평가 항목·표본 수·합격 점수는 임상기관이 정하며 이 코드에 하드코딩하지 않는다.
    대신 ``EvaluationRecord``가 평가셋·평가자·승인 참조를 모두 갖고 정확한 버전과
    일치할 때만 활성화한다. 레코드가 없거나 거부된 경우 기본적으로 차단한다.
    """

    def __init__(self, records: Iterable[EvaluationRecord] = ()) -> None:
        self._records = tuple(records)

    def require_approved(self, version: ChatVersion) -> None:
        """운영 후보 버전에 유효한 활성화 승인 기록이 있는지 확인한다.

        Args:
            version: 호출하려는 Provider·모델·프롬프트·지식 버전 조합이다.

        Raises:
            EvaluationNotApprovedError: 승인 레코드가 없거나 필수 추적 정보가 없는
                경우 발생한다. 호출자는 Provider 요청을 시작하면 안 된다.
        """
        for record in self._records:
            if (
                record.version == version
                and record.approved_for_activation
                and record.evaluation_set_id.strip()
                and record.evaluator_id.strip()
                and record.approval_reference.strip()
            ):
                return
        raise EvaluationNotApprovedError("clinical chat version is not approved for activation")
