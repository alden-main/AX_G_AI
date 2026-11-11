"""Field tables of ``docs/assessment_schema.md``, generated from ``schema`` so the document cannot drift.

    uv run python -m ax_g_ai.services.clinical_recommendation.schema_doc            # rewrite the tables
    uv run python -m ax_g_ai.services.clinical_recommendation.schema_doc --check    # exit 1 when out of date
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
    "AssessmentPayload", "RecommendationItem", "LabInterpretation", "MappingIssue", "MedicationAlerts", "DurAlert", "DurProduct",
    "DurNotice", "DurUnmapped", "DurSource", "DoseAssessment", "DoseDrug", "DosePrescription", "DoseCheck", "DoseCalculation",
    "DoseRenal", "RenalBand", "DoseConditionRule", "LabeledValue", "UnmatchedDoseDrug", "Diagnosis", "TestEntry", "MissingInput",
    "Source",
)
_BASIC = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}


def type_label(annotation: Any) -> str:
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
    head, rest = document.split(START, 1)
    _old, tail = rest.split(END, 1)
    return f"{head}{START}\n\n{tables()}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
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
