import unittest

from fastapi.testclient import TestClient

from ax_g_ai.adapters.openai import ProviderAnswer
from ax_g_ai.api import create_app
from ax_g_ai.runtime import PatientContextStore
from ax_g_ai.services.clinical_chat import ClinicalChatService


class Provider:
    def __init__(self):
        self.requests = []
        self.fail = False

    def respond(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError('failed')
        return ProviderAnswer('상태: 혈당 120. 처방: 기록 없음.')


class EmrSummaryTest(unittest.TestCase):
    def test_one_shot_summary_and_failures(self):
        provider = Provider()
        service = ClinicalChatService(None, None, provider, None, None, None)
        client = TestClient(create_app(service, PatientContextStore(service)))
        payload = {'patient': {'patientId': 'p1', 'patientName': '비밀이름'}, 'bloodSugarList': [{'measuredAt': '2026-09-22 10:30:00', 'glucoseValue': 120}]}
        response = client.post('/internal/v1/emr-summary', json={'patient_id': 'p1', 'emr_payload': payload})
        assert response.status_code == 200
        assert response.json()['answer'].startswith('상태:')
        assert response.json()['emr_updated_at'] == '2026-09-22T10:30:00+09:00'
        assert len(provider.requests) == 1
        assert '비밀이름' not in provider.requests[0].input
        assert client.post('/internal/v1/clinical-chat', json={'patient_id': 'p1', 'question': '질문'}).status_code == 404
        assert client.post('/internal/v1/emr-summary', json={'patient_id': 'p2', 'emr_payload': payload}).status_code == 403
        assert client.post('/internal/v1/emr-summary', json={'patient_id': 'p1', 'emr_payload': {}}).status_code == 502
        assert len(provider.requests) == 1
        provider.fail = True
        response = client.post('/internal/v1/emr-summary', json={'patient_id': 'p1', 'emr_payload': payload})
        assert response.status_code == 502
        assert response.json()['code'] == 'AI_PROVIDER_UNAVAILABLE'
