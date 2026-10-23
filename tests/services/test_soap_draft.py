from __future__ import annotations

from datetime import date
import json
from unittest import TestCase

from ax_g_ai.adapters.openai import HttpResponse, OpenAIResponsesClient
from ax_g_ai.services.soap_draft import PreviousSoap, SoapDraftRequest, SoapDraftService, compose_soap_draft_input


class Transport:
    def __init__(self) -> None:
        self.body = json.dumps({
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({
                "subjective": "S", "objective": "O", "assessment": "A", "plan": "P",
                "data_limitations": ["제한"], "review_notice": "검토 필요",
                "input_needed": [], "verification_needed": [], "evidence": ["EMR 혈당"],
            }, ensure_ascii=False)}]}]
        }, ensure_ascii=False).encode()
        self.request: dict[str, object] | None = None

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse:
        self.request = json.loads(body)
        return HttpResponse(200, "application/json", self.body)


class SoapDraftServiceTest(TestCase):
    def test_generates_structured_draft_with_only_allowed_snapshot_fields(self) -> None:
        transport = Transport()
        service = SoapDraftService(OpenAIResponsesClient(transport, "secret", "model", 5))
        draft = service.draft(
            request=SoapDraftRequest("patient-1", "admission_initial", date(2026, 9, 23), admission_record="입원 기록"),
            snapshot={"blood_sugar": [{"value": 120, "measured_at": "2026-09-23", "patient_name": "제외"}], "patient": {"name": "제외"}},
        )
        self.assertEqual(draft.assessment, "A")
        assert transport.request is not None
        self.assertEqual(transport.request["text"]["format"]["name"], "soap_inpatient_draft")  # type: ignore[index]
        self.assertNotIn("제외", str(transport.request))

    def test_next_day_input_contains_structured_previous_soap(self) -> None:
        rendered = compose_soap_draft_input(
            SoapDraftRequest("patient-1", "next_day", date(2026, 9, 23), previous_soap=PreviousSoap(date(2026, 9, 22), "S", "O", "A", "P")),
            {},
        )
        self.assertIn('"recorded_on":"2026-09-22"', rendered)
        self.assertNotIn("patient-1", rendered)
