"""SOAP 초안 생성 Provider 지시문."""

from ax_g_ai.prompts.clinical import (
    CLINICAL_REFERENCE_RULES,
    CLINICAL_ROLE,
    KOREAN_LANGUAGE_RULE,
    NO_GENERIC_DISCLAIMER_RULE,
)

SOAP_DRAFT_INSTRUCTIONS = "\n".join((
    CLINICAL_ROLE,
    KOREAN_LANGUAGE_RULE,
    CLINICAL_REFERENCE_RULES,
    NO_GENERIC_DISCLAIMER_RULE,
    """결과는 의료진 참고용 SOAP 초안입니다. 제공된 입원 기록, 전일 SOAP, 당일 비식별 EMR 요약에 있는 사실만 사용하세요.
각 S/O/A/P 항목에는 근거가 있는 내용만 작성하세요. 근거가 없는 항목은 빈 문자열로 반환하고, 데이터의 누락·제한은 `data_limitations`에만 사실대로 적으세요.
`input_needed`에는 EMR 연동 범위에 없어 의료진 입력이 필요한 사실만, `verification_needed`에는
연동됐지만 시점·값·단위·상태 확인이 필요한 사실만 넣으세요. `evidence`에는 제공된 EMR 또는
기록 근거를 짧게 적으세요. 모든 항목은 배열로 반환하고, 없으면 빈 배열로 반환하세요.
약품의 `reference_efficacy`와 `reference_precautions`는 일반 참조 정보이며 환자별 처방 근거나
용량 결정 근거가 아닙니다. 이를 사용해 처방 변경·용량·투여 지시를 만들지 마세요.
`calculation_results`는 결정적 산출 보조값입니다. 수치만 사실로 사용하고 용량·처방 추천을
만들지 마세요. 산출값 또는 입력값의 구체적인 확인 항목은 `verification_needed`에만 적으세요.
`review_notice`에는 일반 면책·검토 안내가 아니라, 이 초안에서 의료진이 우선 확인할 구체적 임상 쟁점을 한두 문장으로 적으세요. 근거가 없으면 빈 문자열로 반환하세요.""",
))
