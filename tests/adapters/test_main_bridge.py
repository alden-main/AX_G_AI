from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import unittest

from ax_g_ai.adapters.emr import EmrAdapterError
from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper


class RawEmrPayloadMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = Path(__file__).parents[2] / "emr_test.json"
        self.payload = json.loads(fixture.read_text(encoding="utf-8"))
        self.mapper = RawEmrPayloadMapper()

    def test_maps_known_lists_and_excludes_null_values_and_identifiers(self) -> None:
        context = self.mapper.map("pid-4356", self.payload)

        self.assertEqual(len(context.data["blood_pressure"]), 2)
        self.assertNotIn("timing_type", context.data["blood_sugar"][0])
        self.assertEqual(context.data["blood_sugar"][2]["timing_type"], "식전")
        self.assertEqual(context.data["lab_results"], [{"test_date": "2026-09-04T09:00:00+09:00", "value": "6.8", "exam_code": "HbA1c", "item_name": "당화혈색소"}])
        self.assertEqual(context.updated_at.isoformat(), "2026-09-04T09:00:00+09:00")
        rendered = json.dumps(context.data, ensure_ascii=False)
        for forbidden in ("홍길동", "doc-5567", "임꺽정", "instructions", "하루 3번"):
            self.assertNotIn(forbidden, rendered)

    def test_null_or_missing_lists_are_empty_and_server_time_is_used_without_dates(self) -> None:
        context = self.mapper.map("patient-1", {"patient": {"patientId": "patient-1"}, "bloodPressureList": None}, prepared_at=datetime.fromisoformat("2026-09-22T10:30:00+09:00"))

        self.assertEqual(context.data, {"blood_pressure": [], "blood_sugar": [], "oxygen_saturation": [], "lab_results": [], "prescriptions": []})
        self.assertEqual(context.updated_at.isoformat(), "2026-09-22T10:30:00+09:00")

    def test_patient_mismatch_is_rejected_before_context_is_created(self) -> None:
        with self.assertRaisesRegex(EmrAdapterError, "patient does not match"):
            self.mapper.map("other", self.payload)


if __name__ == "__main__":
    unittest.main()
