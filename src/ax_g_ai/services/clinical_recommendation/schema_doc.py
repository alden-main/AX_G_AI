"""``docs/assessment_schema.md``의 필드 표를 ``schema``에서 생성해 문서가 코드와 어긋나지 않게 하는 도구.

설계상 위치: 임상 추천 패키지의 문서 생성 도구(개발용 명령). 개발자가 직접 실행하고, 테스트가 ``render`` 결과로
문서가 최신인지 확인한다.

역할: ``MODELS``에 나열한 Pydantic 모델마다 docstring 첫 줄(요약)과 필드 표(필드, 타입, 필수 여부, 설명)를 만들어
문서의 ``<!-- schema-tables:start -->``와 ``<!-- schema-tables:end -->`` 사이를 바꾼다.

    uv run python -m ax_g_ai.services.clinical_recommendation.schema_doc            # 표 다시 쓰기
    uv run python -m ax_g_ai.services.clinical_recommendation.schema_doc --check    # 최신이 아니면 1로 종료
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from ax_g_ai.services.clinical_recommendation import schema

DOC = Path(__file__).resolve().parents[4] / "docs" / "assessment_schema.md"
START, END = "<!-- schema-tables:start -->", "<!-- schema-tables:end -->"
MODELS = (
    "AssessmentPayload", "FindingConflict", "RecommendationItem", "LabInterpretation", "MappingIssue", "MedicationAlerts", "DurAlert", "DurProduct",
    "DurNotice", "DurUnmapped", "DurSource", "DoseAssessment", "DoseDrug", "DosePrescription", "DoseCheck", "DoseCalculation",
    "DoseRenal", "RenalBand", "DoseConditionRule", "LabeledValue", "UnmatchedDoseDrug", "Diagnosis", "TestEntry", "MissingInput",
    "Source",
)
_BASIC = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}


def type_label(annotation: Any) -> str:
    """타입 주석을 문서 표에 쓸 타입 이름으로 바꾼다.

    역할: 기본 타입은 JSON 이름(string, integer …)으로, ``Literal``은 값 나열로, 합집합은 ``|``로 잇되 int|float는
    number로 합치고, 목록·dict·모델(문서 안 링크)도 재귀로 표기한다.
    입력:
        annotation: 필드의 타입 주석.
    출력: Markdown 표용 타입 문자열(``|``는 이스케이프).
    """
    if annotation in _BASIC:
        return _BASIC[annotation]
    if annotation is Any:
        return "any"
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return " \\| ".join(f'`"{value}"`' for value in args)
    if origin in (Union, types.UnionType):
        labels = list(dict.fromkeys(type_label(arg) for arg in args))
        return " \\| ".join(["number", *labels[2:]] if labels[:2] == ["integer", "number"] else labels)
    if origin is list:
        inner = type_label(args[0])
        return f"({inner})[]" if "\\|" in inner else f"{inner}[]"
    if origin is dict:
        return "object" if args[1] is Any else f"object<{type_label(args[1])}>"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return f"[`{annotation.__name__}`](#{annotation.__name__.lower()})"
    return str(annotation)


def tables() -> str:
    """``MODELS`` 순서대로 모델별 제목·요약·필드 표 Markdown을 만든다.

    역할: 요약은 모델 docstring의 첫 줄(겹 백틱을 홑 백틱으로 바꿈)이고, 추가 키를 허용하는 모델에는 안내 문장을 붙인다.
    입력: 없음.
    출력: 모든 표를 이은 Markdown 문자열.
    """
    parts: list[str] = []
    for name in MODELS:
        model: type[BaseModel] = getattr(schema, name)
        parts.append(f"### `{name}`\n")
        summary = (model.__doc__ or "").strip().splitlines()
        if summary:
            parts.append(summary[0].replace("``", "`") + "\n")
        if model.model_config.get("extra") == "allow":
            parts.append("규칙 레코드의 다른 필드(`action`, `alert`, `dose` 등)도 그대로 함께 온다.\n")
        parts.append("| 필드 | 타입 | 필수 | 설명 |\n|---|---|:-:|---|")
        for field, info in model.model_fields.items():
            parts.append(f"| `{field}` | {type_label(info.annotation)} | {'✓' if info.is_required() else ''} | {info.description or ''} |")
        parts.append("")
    return "\n".join(parts)


def render(document: str) -> str:
    """문서의 시작·끝 표시 사이를 새로 생성한 표로 바꾼 문서 전체를 돌려준다."""
    head, rest = document.split(START, 1)
    _old, tail = rest.split(END, 1)
    return f"{head}{START}\n\n{tables()}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점: 문서 표를 다시 쓰거나(기본) 최신인지 확인한다(``--check``).

    역할: 현재 문서를 읽어 표를 다시 만들고, ``--check``이면 달라졌을 때 표준 오류에 안내를 쓰고 1을 돌려주며,
    아니면 문서 파일에 쓴다.
    입력:
        argv: 명령행 인자. ``None``이면 ``sys.argv``.
    출력: 종료 코드(0 성공, 1 문서가 최신이 아님).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 when the document tables are out of date")
    args = parser.parse_args(argv)
    current = DOC.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        if updated != current:
            print(f"{DOC} is out of date: run python -m ax_g_ai.services.clinical_recommendation.schema_doc", file=sys.stderr)
            return 1
        return 0
    DOC.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
