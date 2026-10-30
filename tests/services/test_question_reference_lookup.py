import sqlite3
import tempfile
import unittest
from pathlib import Path

from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup
from tests.services.test_clinical_chat import RecordingProvider


class QuestionReferenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_and_both_chat_paths_use_explicit_named_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('''
                    CREATE TABLE diseases (code TEXT, name_ko TEXT);
                    CREATE TABLE medicines (kd_code TEXT, name_ko TEXT, efficacy TEXT, precautions TEXT);
                    CREATE TABLE metadata (key TEXT, value TEXT);
                    INSERT INTO diseases VALUES ('E11', '당뇨병');
                    INSERT INTO medicines VALUES ('1', '시험약', '일반 효능', '일반 주의');
                    INSERT INTO medicines VALUES ('2', '중복약', '효능', '주의');
                    INSERT INTO medicines VALUES ('2', '중복약', '효능', '주의');
                    INSERT INTO metadata VALUES ('medicine_source', 'medicine.pkl');
                ''')
            lookup = SqliteReferenceLookup(path)
            refs = lookup.search_question('당뇨병과 시험약, 중복약을 설명해줘')
            self.assertEqual([r['name'] for r in refs], ['당뇨병', '시험약'])
            self.assertEqual(refs[1]['source'], 'medicine.pkl')
            self.assertEqual(lookup.search_question('관련 없는 질문'), [])
            provider = RecordingProvider()
            service = ClinicalChatService(provider, reference_lookup=lookup)
            await service.answer_prepared_snapshot(question='시험약을 설명해줘', snapshot={'blood_sugar': []})
            self.assertIn('일반 효능', provider.requests[0].input)
            self.assertIn('진단 기준이 아닌', provider.requests[0].input)
            [delta async for delta in service.stream_prepared_snapshot(question='시험약을 설명해줘', snapshot={})]
            self.assertIn('일반 효능', provider.requests[1].input)

    def test_longest_names_are_preferred_and_results_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE diseases (code TEXT, name_ko TEXT)')
                db.execute('CREATE TABLE medicines (kd_code TEXT, name_ko TEXT, efficacy TEXT, precautions TEXT)')
                db.executemany('INSERT INTO diseases VALUES (?, ?)', [
                    ('E10', '당뇨'), ('E11', '2형 당뇨병'), ('E12', '당뇨병'), ('X', '병'),
                ] + [(f'Z{i}', f'질환{i}') for i in range(10)])
            lookup = SqliteReferenceLookup(path)

            names = [r['name'] for r in lookup.search_question('2형 당뇨병 질환1 질환2 질환3 질환4')]

            self.assertEqual(names[:3], ['2형 당뇨병', '당뇨병', '질환1'])
            self.assertEqual(len(names), 5)
            self.assertNotIn('병', names)

    def test_missing_database_is_optional(self):
        self.assertEqual(SqliteReferenceLookup(Path('/nonexistent/reference.sqlite')).search_question('당뇨병'), [])
