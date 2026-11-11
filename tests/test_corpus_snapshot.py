"""One corpus snapshot per request, swapped as a whole; rule facts re-mapped for a newer release."""

from __future__ import annotations

import asyncio
import copy
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper, rule_source
from ax_g_ai.api import create_app
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot
from ax_g_ai.services.patient_context_store import PatientContextStore
from tests.services.test_clinical_recommendation import RULES, recommender, source
from tests.test_clinical_recommendation_api import ScriptedProvider, payload


class SwitchableCorpus:
    def __init__(self, snapshot: CorpusSnapshot) -> None:
        self.current = snapshot

    def close(self) -> None:
        return None


class RuleSourceTest(unittest.TestCase):
    def test_rule_facts_from_the_stored_source_equal_those_from_the_raw_emr(self) -> None:
        engine = recommender()
        variants = [source()]
        empty = copy.deepcopy(source())
        empty["medicineDtailList"], empty["diagnosisList"] = [], []  # "recorded: none" must stay distinct from absent
        variants.append(empty)
        sparse = copy.deepcopy(source())
        sparse["medicineDtailList"] = [{"medCode": None, "medName": None}, "not an object"]
        sparse.pop("labResultList", None)
        sparse["patient"] = {"patientId": "pid-1", "name": "홍길동"}
        variants.append(sparse)
        for raw in variants:
            with self.subTest(keys=sorted(raw)):
                self.assertEqual(engine.prepare(rule_source(raw)), engine.prepare(raw))

    def test_rule_source_keeps_no_identifiers(self) -> None:
        raw = copy.deepcopy(source())
        raw["patient"] = {"patientId": "pid-1", "name": "홍길동"}
        raw["medicineDtailList"] = [{"medCode": "X", "medName": "약", "prescriberName": "의사", "patientId": "pid-1"}]
        kept = rule_source(raw)
        self.assertNotIn("patient", kept)
        self.assertEqual(kept["medicineDtailList"], [{"medCode": "X", "medName": "약"}])


class SnapshotApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = ScriptedProvider()
        evidence = InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)
        self.corpus = SwitchableCorpus(CorpusSnapshot(evidence, recommender(), release_id=1))
        service = ClinicalChatService(self.provider)  # evidence comes from the snapshot only
        self.store = PatientContextStore()
        self.api = TestClient(create_app(service, RawEmrPayloadMapper(), self.store, corpus=self.corpus))

    def test_prepared_context_records_its_release_and_is_remapped_for_a_newer_one(self) -> None:
        self.api.post("/internal/v1/patient-context", json=payload())
        self.assertEqual(asyncio.run(self.store.get("pid-1")).rule_release, 1)
        newer = recommender()
        self.corpus.current = CorpusSnapshot(self.corpus.current.evidence, newer, release_id=2)
        with patch.object(newer, "prepare", wraps=newer.prepare) as remap:
            response = self.api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
            self.assertEqual(response.status_code, 200)
            self.api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
        self.assertEqual(remap.call_count, 2)  # mapped with release 2's rules (per request; the stored context is not rewritten)

    def test_same_release_reuses_the_prepared_facts(self) -> None:
        self.api.post("/internal/v1/patient-context", json=payload())
        engine = self.corpus.current.recommender
        with patch.object(engine, "prepare", wraps=engine.prepare) as remap:
            self.api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
        remap.assert_not_called()

    def test_evidence_is_searched_in_the_request_snapshot(self) -> None:
        self.api.post("/internal/v1/patient-context", json=payload())
        response = self.api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "만성 콩팥병 추천 검사"})
        self.assertTrue(response.json().get("evidence"))
        self.corpus.current = CorpusSnapshot(InMemoryEvidenceIndex([]), None, release_id=3)
        response = self.api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "만성 콩팥병 추천 검사"})
        self.assertNotIn("evidence", response.json())  # the new (empty) snapshot, not a retriever kept in the service


class ConcurrentCallsTest(unittest.TestCase):
    def test_note_extraction_and_evidence_search_run_at_the_same_time(self) -> None:
        overlap = {"seen": False}

        class Service(ClinicalChatService):
            loop: asyncio.AbstractEventLoop
            searching = threading.Event()

            async def extract_findings(self, *, note, vocabulary):
                # Completes only if the evidence search is running meanwhile (sequential calls would time out).
                await asyncio.get_running_loop().run_in_executor(None, self.searching.wait, 2)
                overlap["seen"] = self.searching.is_set()
                return await super().extract_findings(note=note, vocabulary=vocabulary)

            def find_evidence(self, question, retriever=None):
                self.searching.set()
                return super().find_evidence(question, retriever)

        evidence = InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)
        api = TestClient(create_app(Service(ScriptedProvider(), evidence=evidence), RawEmrPayloadMapper(), PatientContextStore(),  # type: ignore[arg-type]
                                    recommender()))
        api.post("/internal/v1/patient-context", json=payload())
        response = api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "clinical_note": "혈변"})
        self.assertTrue(overlap["seen"])
        self.assertEqual(response.json()["assessment"]["findings_used"].get("hematochezia"), True)


if __name__ == "__main__":
    unittest.main()
