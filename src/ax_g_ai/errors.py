"""API가 안전한 오류 코드로 변환하는 애플리케이션 예외.

설계상 위치: 서비스·어댑터 계층과 API 계층 사이의 공용 예외 정의. 서비스·어댑터가 발생시키고
``api.py``의 ``_ERROR_RESPONSES``가 예외 타입별로 HTTP 상태·오류 코드·안내 문구에 대응시킨다.

역할: 예외 메시지에는 EMR 원문·질문·Provider 응답을 넣지 않는다. API 계층은 예외 타입만으로
HTTP 상태와 오류 코드를 정한다.
"""

from __future__ import annotations


class PatientContextNotReadyError(LookupError):
    """채팅 전에 환자 Snapshot이 준비되지 않았거나 보관 기간이 지났을 때 발생한다.

    역할: API에서 404 ``PATIENT_CONTEXT_NOT_READY``로 바뀐다.
    """


class PatientContextUnavailableError(RuntimeError):
    """EMR payload 검증이 실패했을 때 발생한다.

    역할: API에서 502 ``EMR_UNAVAILABLE``로 바뀐다. ``failure``는 로그에 남겨도 되는 스키마 진단 코드이며
    요청 내용을 담지 않는다.
    """

    def __init__(self, failure: str = "emr_payload:invalid") -> None:
        super().__init__(failure)
        self.failure = failure


class PatientMismatchError(PermissionError):
    """요청 환자와 전달된 EMR의 환자가 다를 때 발생한다.

    역할: API에서 403 ``PATIENT_MISMATCH``로 바뀐다.
    """


class ClinicalChatProviderUnavailableError(RuntimeError):
    """AI Provider가 설정되지 않았거나 호출·응답 검증이 실패했을 때 발생한다.

    역할: API에서 502 ``AI_PROVIDER_UNAVAILABLE``로 바뀐다(스트리밍 중이면 SSE ``error`` 이벤트).
    """


class RecommendationUnavailableError(RuntimeError):
    """말뭉치 규칙(추천 진단·검사)이 설정되지 않았거나 불러온 규칙이 없을 때 발생한다.

    역할: API에서 503 ``RECOMMENDATION_UNAVAILABLE``로 바뀐다.
    """
