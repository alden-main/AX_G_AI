from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import unittest

from ax_g_ai.adapters.emr import EmrAdapterError
from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup


class RawEmrPayloadMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = Path(__file__).parents[2] / "emr_test.json"
        self.payload = json.loads(fixture.read_text(encoding="utf-8"))
        self.mapper = RawEmrPayloadMapper()

    def test_maps_known_lists_and_excludes_null_values_and_identifiers(self) -> None:
        context = self.mapper.map("pid-4356", self.payload)

        self.assertEqual(len(context.data["blood_pressure"]), 3)
        self.assertEqual(context.data["blood_pressure"][-1]["dbp"], 83)
        self.assertNotIn("sbp", context.data["blood_pressure"][-1])
        self.assertNotIn("timing_type", context.data["blood_sugar"][0])
        self.assertEqual(context.data["blood_sugar"][2]["timing_type"], "식전")
        self.assertEqual(context.data["lab_results"], [{"test_date": "2026-09-04T09:00:00+09:00", "value": "6.8", "exam_code": "HbA1c", "item_name": "당화혈색소"}])
        self.assertEqual(context.updated_at.isoformat(), "2026-09-04T09:00:00+09:00")
        rendered = json.dumps(context.data, ensure_ascii=False)
        for forbidden in ("홍길동", "doc-5567", "임꺽정", "instructions", "하루 3번"):
            self.assertNotIn(forbidden, rendered)

    def test_null_or_missing_lists_are_empty_and_server_time_is_used_without_dates(self) -> None:
        context = self.mapper.map("patient-1", {"patient": {"patientId": "patient-1"}, "bloodPressureList": None}, prepared_at=datetime.fromisoformat("2026-09-22T10:30:00+09:00"))

        self.assertEqual(context.data, {"blood_pressure": [], "blood_sugar": [], "oxygen_saturation": [], "lab_results": [], "prescriptions": [], "diagnoses": [], "calculation_results": [], "calculation_input_needed": ["BMI·체표면적 계산에는 체중(kg)과 키(cm) 입력이 필요합니다.", "Cockcroft–Gault CrCl 계산에는 연령, 성별(male/female), 체중(kg), 혈청 크레아티닌(mg/dL) 입력이 필요합니다."]})
        self.assertEqual(context.updated_at.isoformat(), "2026-09-22T10:30:00+09:00")

    def test_patient_mismatch_is_rejected_before_context_is_created(self) -> None:
        with self.assertRaisesRegex(EmrAdapterError, "patient does not match"):
            self.mapper.map("other", self.payload)

    def test_maps_the_bridge_data_envelope_and_additional_deidentified_clinical_fields(self) -> None:
        context = self.mapper.map("patient-1", {
            "data": {
                "patient": {"patientId": "patient-1"},
                "bloodPressureList": [
                    {"measuredAt": "2026-08-13 19:00:00", "sbp": "112", "dbp": None},
                    {"measuredAt": "2026-08-13 18:00:00", "sbp": None, "dbp": "60"},
                ],
                "diagnosisList": [{"kcdCode": "E11.9", "kcdName": "2형 당뇨병"}],
                "unrecognizedList": [{"value": "활용할 임상 정보", "patientId": "patient-1"}],
            }
        })

        self.assertEqual(context.data["blood_pressure"], [
            {"measured_at": "2026-08-13T19:00:00+09:00", "unit": "mmHg", "sbp": "112"},
            {"measured_at": "2026-08-13T18:00:00+09:00", "unit": "mmHg", "dbp": "60"},
        ])
        self.assertEqual(context.data["diagnoses"], [{"code": "E11.9", "name_ko": "2형 당뇨병"}])
        self.assertEqual(context.data["additional_clinical_data"], {"unrecognizedList": [{"value": "활용할 임상 정보"}]})

    def test_enriches_a_prescription_from_the_optional_sqlite_reference_data(self) -> None:
        from tempfile import TemporaryDirectory
        import sqlite3

        with TemporaryDirectory() as directory:
            database = Path(directory) / "reference.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """CREATE TABLE medicines (
                    source_kd_code TEXT PRIMARY KEY, kd_code TEXT NOT NULL,
                    name_ko TEXT NOT NULL, efficacy TEXT, usage TEXT, precautions TEXT
                    )"""
                )
                connection.execute(
                    """INSERT INTO medicines VALUES
                    ('100', '100', '참조 약품', '참조 효능', '참조 용법', '참조 주의사항')"""
                )
                connection.execute(
                    "CREATE TABLE diseases (code TEXT PRIMARY KEY, name_ko TEXT NOT NULL)"
                )
                connection.execute("INSERT INTO diseases VALUES ('A000', '검사용 질환')")
            mapper = RawEmrPayloadMapper(SqliteReferenceLookup(database))
            context = mapper.map("patient-1", {
                "patient": {"patientId": "patient-1"},
                "medicineDtailList": [{"medCode": "100", "dosage": 1}],
                "diagnosisList": [{"diagnosisCode": "A000"}],
            })

        prescription = context.data["prescriptions"][0]
        self.assertEqual(prescription["med_name"], "참조 약품")
        self.assertEqual(prescription["reference_efficacy"], "참조 효능")
        self.assertEqual(prescription["reference_precautions"], "참조 주의사항")
        self.assertNotIn("reference_usage", prescription)
        self.assertEqual(context.data["diagnoses"], [{"code": "A000", "name_ko": "검사용 질환"}])


if __name__ == "__main__":
    unittest.main()
