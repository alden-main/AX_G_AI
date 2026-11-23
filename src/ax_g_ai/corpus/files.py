"""생성된 말뭉치(``text_data/corpus``)의 파일 이름 모음: 빌드·적재기·런타임 사이의 계약.

설계상 위치: 말뭉치 계약 계층(``corpus`` 패키지). 말뭉치 빌드, 적재기, 근거 색인(``clinical_evidence``)과
규칙 말뭉치 로더가 파일 이름을 이 상수로만 참조한다.

역할: 근거 문단·규칙·개념 파일 이름과, 저장된 문단 벡터를 구분하는 기본 임베딩 모델 id를 정의한다.
"""

from __future__ import annotations

CHUNKS_FILE = "chunks.jsonl"
DOSE_FILE = "dose_rules.jsonl"
# 런타임이 평가하는 규칙 레코드(2단계, 용량 규칙은 3단계).
RULE_FILES = ("diagnosis.jsonl", "workup.jsonl", "referral.jsonl", "monitoring.jsonl", "lab_thresholds.jsonl", DOSE_FILE)
# 적재기가 저장하는 모든 레코드 파일. 런타임이 평가하지 않는 레코드(치료, 공식)도 포함한다.
RECORD_FILES = (
    "diagnosis.jsonl", "workup.jsonl", "treatment.jsonl", DOSE_FILE, "lab_thresholds.jsonl",
    "formulas.jsonl", "monitoring.jsonl", "referral.jsonl",
)
CONCEPT_FILES = ("test_concepts.jsonl", "drug_concepts.jsonl", "drug_classes.jsonl")

# 저장된 문단 벡터의 키가 되는 모델 id(``corpus/embeddings/<id>.jsonl``, ``evidence_embedding_1024.model_id``).
DEFAULT_EMBEDDING_MODEL = "kure-v1"
