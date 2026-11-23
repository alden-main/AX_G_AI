"""DUR snapshot, checker and PostgreSQL store tests.

The PostgreSQL tests need a disposable pgvector/PostgreSQL database (they create and drop the
``dur`` schema); see tests/corpus_store/test_postgres_store.py for the docker command::

    AX_G_TEST_DATABASE_URL=postgresql://postgres:itpass@127.0.0.1:55499/postgres uv run python -m unittest tests.services.test_dur_check
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ax_g_ai.dur_store.snapshot import DurDataError, build, main, read_snapshot
from ax_g_ai.services.dur_check import DurChecker, DurTables, DurUnavailableError

REPO_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "dur"
TEST_DATABASE_URL = os.getenv("AX_G_TEST_DATABASE_URL")

FILES = {
    "의약품안전사용서비스(DUR)_병용금기 품목리스트 2026.6.csv": (
        ["성분명A", "성분코드A", "제품코드A", "제품명A", "업체명A", "급여여부A", "성분명B", "성분코드B", "제품코드B", "제품명B", "업체명B", "급여여부B", "고시번호", "고시일자", "상세정보", "비고"],
        [
            ["clarithromycin", "136901ATB", "111111111", "클래리정", "A사", "급여", "simvastatin", "227801ATB", "222222222", "심바정", "B사", "급여", "2020", "2020-01-01", "근육병증의 위험 증가 가능", ""],
            ["simvastatin", "227801ATB", "222222223", "심바정40", "C사", "급여", "clarithromycin", "136901ATB", "111111112", "클래리정500", "D사", "급여", "2020", "2020-01-01", "근육병증의 위험 증가 가능", ""],
            ["methotrexate", "193201ATB", "333333333", "엠티엑스정", "E사", "급여", "trimethoprim", "246501ATB", "444444444", "셉트린정", "F사", "급여", "2021", "2021-01-01", "골수억제", "1주에 MTX 15mg 이상 투여시 병용금기"],
        ],
    ),
    "의약품안전사용서비스(DUR)_연령금기 품목리스트 2026.6.csv": (
        ["성분명", "성분코드", "제품코드", "제품명", "업체명", "특정연령", "특정연령단위", "연령처리조건", "고시번호", "고시일자", "상세정보", "급여여부"],
        [["levofloxacin", "182901ATB", "555555555", "레보정", "G사", "18", "세 (3)", "미만 (1)", "2007", "2007-04-11", "18세 미만 금기(관절병증)", "급여"]],
    ),
    "의약품안전사용서비스(DUR)_임부금기 품목리스트 2026.6.csv": (
        ["성분명", "성분코드", "제품코드", "제품명", "업체명", "고시일자", "고시번호", "금기등급", "상세정보", "급여여부"],
        [["simvastatin", "227801ATB", "222222222", "심바정", "B사", "2016-12-30", "2016", "1", "임부 투여금기", "급여"]],
    ),
    "의약품안전사용서비스(DUR)_노인주의 품목리스트 2026.6.csv": (
        ["성분명", "성분코드", "제품코드", "제품명", "업소명", "공고일자", "공고번호", "약품상세정보", "비고", "급여여부"],
        [["diazepam", "141901ATB", "666666666", "디아제팜정", "H사", "2020-09-24", "2020", "낙상 위험 증가", "", "급여"]],
    ),
    "의약품안전사용서비스(DUR)_노인주의(해열진통소염제) 품목리스트 2026.6.csv": (
        ["성분코드", "성분명", "제품코드", "제품명", "업소명", "약품상세정보", "급여여부"],
        [["100901ACH", "aceclofenac", "777777777", "에이서캡슐", "I사", "고령자 위장관 출혈 위험", "급여"]],
    ),
}


def write_fixture(directory: Path) -> None:
    for name, (header, rows) in FILES.items():
        with (directory / name).open("w", encoding="cp949", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


def fixture_snapshot() -> Path:
    """CSV fixture → snapshot folder in a temp dir (caller removes ``path.parent``)."""
    tmp = Path(tempfile.mkdtemp())
    csv_dir = tmp / "csv"
    csv_dir.mkdir()
    write_fixture(csv_dir)
    build(csv_dir, tmp / "dur")
    return tmp / "dur"


def checker_from_snapshot(directory: Path) -> DurChecker:
    snapshot = read_snapshot(directory)
    return DurChecker(DurTables.from_rows(snapshot.source, snapshot.rows))


class DurSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot_dir = fixture_snapshot()
        self.addCleanup(shutil.rmtree, self.snapshot_dir.parent)
        self.snapshot = read_snapshot(self.snapshot_dir)
        self.checker = checker_from_snapshot(self.snapshot_dir)

    def test_rules_are_deduplicated_to_ingredient_level(self) -> None:
        counts = {name: len(rows) for name, rows in self.snapshot.rows.items()}
        self.assertEqual(counts["pair_rule"], 2)  # the reversed clarithromycin–simvastatin row collapses
        self.assertEqual(counts["product"], 9)
        self.assertEqual(self.snapshot.source["license"], "KOGL Type 1 (공공누리 제1유형 — 출처표시)")
        self.assertEqual(len(self.snapshot.manifest["inputs"]["pair"]["sha256"]), 64)

    def test_rebuild_of_the_same_csv_keeps_the_dataset_identity(self) -> None:
        again = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, again)
        build(self.snapshot_dir.parent / "csv", again / "dur")
        self.assertEqual(read_snapshot(again / "dur").content_sha256, self.snapshot.content_sha256)

    def test_tampered_or_missing_snapshot_is_rejected(self) -> None:
        path = self.snapshot_dir / "pair_rule.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("근육병증", "근육통"), encoding="utf-8")
        with self.assertRaises(DurDataError):
            read_snapshot(self.snapshot_dir)
        with self.assertRaises(DurDataError):
            read_snapshot(self.snapshot_dir.parent / "nothing")

    def test_pair_age_pregnancy_and_elderly_alerts(self) -> None:
        result = self.checker.check([("111111112", "클래리정500"), ("222222222", "심바정"), ("999999999", "모르는약")],
                                    age_years=40, sex="female", pregnant=True)
        types = [alert.type for alert in result.alerts]
        self.assertEqual(types, ["병용금기", "임부금기"])
        self.assertEqual(result.alerts[0].detail, "근육병증의 위험 증가 가능")
        self.assertEqual(result.alerts[1].grade, "1등급")
        self.assertEqual(result.unmapped[0].med_code, "999999999")
        self.assertEqual(result.source.id, "KR-HIRA-DUR-2026-06")

        self.assertEqual([a.type for a in self.checker.check([("555555555", None)], age_years=15, sex="male", pregnant=None).alerts], ["연령금기"])
        self.assertEqual(self.checker.check([("555555555", None)], age_years=30, sex="male", pregnant=None).alerts, [])
        self.assertEqual(self.checker.check([("555555555", None)], age_years=None, sex=None, pregnant=None).alerts, [])  # age unknown: no guess
        elderly = self.checker.check([("666666666", None), ("777777777", None)], age_years=70, sex="male", pregnant=None).alerts
        self.assertEqual({a.category for a in elderly}, {"노인주의", "노인주의(해열진통소염제)"})

    def test_dose_conditions_are_shown_verbatim_and_pregnancy_unknown_is_a_question(self) -> None:
        pair = self.checker.check([("333333333", None), ("444444444", None)], age_years=50, sex="male", pregnant=None).alerts[0]
        self.assertEqual(pair.condition_note, "1주에 MTX 15mg 이상 투여시 병용금기")
        unknown = self.checker.check([("222222222", None)], age_years=30, sex="female", pregnant=None).alerts
        self.assertEqual((unknown[0].type, unknown[0].status), ("임부금기", "possible"))
        self.assertEqual(self.checker.check([("222222222", None)], age_years=60, sex="female", pregnant=None).alerts, [])
        # Memo and question disagree: confirm whatever the age (unknown sex included), never for a male patient.
        for age, sex in ((60, "female"), (None, None)):
            conflict = self.checker.check([("222222222", None)], age_years=age, sex=sex, pregnant=None, pregnancy_conflict=True).alerts
            self.assertEqual([(a.type, a.status) for a in conflict], [("임부금기", "possible")])
            self.assertIn("서로 다름", conflict[0].missing[0].label)  # type: ignore[index]
        self.assertEqual(self.checker.check([("222222222", None)], age_years=30, sex="male", pregnant=None, pregnancy_conflict=True).alerts, [])
        self.assertEqual(self.checker.check([("222222222", None)], age_years=30, sex="female", pregnant=False).alerts, [])

    def test_missing_files_cli_refuses_to_overwrite_and_empty_tables_are_unavailable(self) -> None:
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty)
        with self.assertRaises(DurDataError):
            build(empty, empty / "out")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--input", str(self.snapshot_dir.parent / "csv"), "--output", str(self.snapshot_dir)]), 1)
        with self.assertRaises(DurUnavailableError):
            DurTables.from_rows({}, {"product": []})


@unittest.skipUnless(TEST_DATABASE_URL, "AX_G_TEST_DATABASE_URL not set (needs a disposable PostgreSQL database)")
class PostgresDurStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        import psycopg

        self.conn = psycopg.connect(TEST_DATABASE_URL, autocommit=True)
        self.conn.execute("DROP SCHEMA IF EXISTS dur CASCADE")
        self.addCleanup(self.conn.close)
        self.addCleanup(lambda: self.conn.execute("DROP SCHEMA IF EXISTS dur CASCADE"))
        self.snapshot_dir = fixture_snapshot()
        self.addCleanup(shutil.rmtree, self.snapshot_dir.parent)

    def test_load_reuse_and_reader_give_the_same_alerts_as_the_snapshot(self) -> None:
        from ax_g_ai.dur_store.loader import load_dur
        from ax_g_ai.dur_store.reader import read_active_tables

        first = load_dur(self.conn, self.snapshot_dir)
        self.assertFalse(first.reused)
        self.assertEqual(first.counts["pair_rule"], 2)
        again = load_dur(self.conn, self.snapshot_dir)
        self.assertEqual((again.dataset_id, again.reused), (first.dataset_id, True))

        from_db = DurChecker(read_active_tables(TEST_DATABASE_URL))
        from_file = checker_from_snapshot(self.snapshot_dir)
        self.assertEqual(from_db.dataset_id, first.dataset_id)
        self.assertEqual(len(from_db), 9)
        meds = [("111111112", "클래리정500"), ("222222222", "심바정"), ("333333333", None), ("444444444", None), ("555555555", None)]
        for kwargs in ({"age_years": 15, "sex": "female", "pregnant": None}, {"age_years": 70, "sex": "male", "pregnant": None}):
            self.assertEqual(from_db.check(meds, **kwargs).model_dump(mode="json"), from_file.check(meds, **kwargs).model_dump(mode="json"))
        self.assertEqual(from_db.source["license"], "KOGL Type 1 (공공누리 제1유형 — 출처표시)")

    def test_new_dataset_activates_and_rollback_restores_the_previous_one(self) -> None:
        from ax_g_ai.dur_store.loader import DurLoadError, activate, load_dur
        from ax_g_ai.dur_store.reader import read_active_tables

        first = load_dur(self.conn, self.snapshot_dir)
        changed = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, changed)
        csv_dir = changed / "csv"
        shutil.copytree(self.snapshot_dir.parent / "csv", csv_dir)
        name = next(p for p in csv_dir.iterdir() if "노인주의 품목" in p.name)
        name.write_text(name.read_text(encoding="cp949").replace("낙상 위험 증가", "낙상·골절 위험 증가"), encoding="cp949")
        build(csv_dir, changed / "dur")
        second = load_dur(self.conn, changed / "dur")
        self.assertNotEqual(second.dataset_id, first.dataset_id)
        details = {rule["detail"] for rules in read_active_tables(TEST_DATABASE_URL).elderly.values() for rule in rules}
        self.assertIn("낙상·골절 위험 증가", details)

        activate(self.conn, first.dataset_id)
        self.assertEqual(read_active_tables(TEST_DATABASE_URL).dataset_id, first.dataset_id)
        active = self.conn.execute("SELECT count(*) FROM dur.dataset WHERE is_active").fetchone()[0]
        self.assertEqual(active, 1)
        with self.assertRaises(DurLoadError):
            activate(self.conn, 9999)

    def test_reader_without_schema_or_active_dataset_is_unavailable(self) -> None:
        from ax_g_ai.dur_store.loader import ensure_schema
        from ax_g_ai.dur_store.reader import read_active_tables

        with self.assertRaises(DurUnavailableError):
            read_active_tables(TEST_DATABASE_URL)
        ensure_schema(self.conn)
        with self.assertRaises(DurUnavailableError):
            read_active_tables(TEST_DATABASE_URL)

    def test_cli_loads_from_the_environment(self) -> None:
        from ax_g_ai.dur_store.loader import main as load_main

        old = os.environ.get("AX_G_CORPUS_DATABASE_URL")
        os.environ["AX_G_CORPUS_DATABASE_URL"] = TEST_DATABASE_URL  # type: ignore[assignment]
        self.addCleanup(lambda: os.environ.__setitem__("AX_G_CORPUS_DATABASE_URL", old) if old else os.environ.pop("AX_G_CORPUS_DATABASE_URL", None))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(load_main(["--snapshot-dir", str(self.snapshot_dir)]), 0)
        self.assertIn("(new)", out.getvalue())


class DurRuntimeTest(unittest.TestCase):
    def test_unreachable_database_disables_dur_without_logging_credentials(self) -> None:
        from ax_g_ai.config import RuntimeSettings
        from ax_g_ai.runtime import _corpus_source

        settings = RuntimeSettings(corpus_database_url="postgresql://nobody:secret@127.0.0.1:1/none")
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            snapshot = _corpus_source(settings).current
        self.assertIsNone(snapshot.recommender)
        self.assertIsNone(snapshot.dur_dataset_id)
        self.assertTrue(any("fallback=files" in line for line in logs.output))
        self.assertFalse(any("secret" in line or "nobody" in line for line in logs.output))
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            _corpus_source(RuntimeSettings())  # no database configured
        self.assertTrue(any("dur_unavailable reason=no_database_url" in line for line in logs.output))


@unittest.skipUnless((REPO_SNAPSHOT / "manifest.json").is_file(), "data/dur snapshot not built")
class RepositoryDurTest(unittest.TestCase):
    def test_real_list_flags_clarithromycin_with_simvastatin(self) -> None:
        snapshot = read_snapshot(REPO_SNAPSHOT)
        checker = DurChecker(DurTables.from_rows(snapshot.source, snapshot.rows))
        rule = next(r for r in snapshot.rows["pair_rule"] if r["name_a"].startswith("clarithromycin") and r["name_b"].startswith("simvastatin"))
        a = next(p["product_code"] for p in snapshot.rows["product"] if p["ingredient_code"] == rule["ingredient_a"])
        b = next(p["product_code"] for p in snapshot.rows["product"] if p["ingredient_code"] == rule["ingredient_b"])
        alerts = checker.check([(a, None), (b, None)], age_years=50, sex="male", pregnant=None).alerts
        self.assertIn("병용금기", [alert.type for alert in alerts])



class QuestionDrugTest(unittest.TestCase):
    """Drugs named in a question are checked as drugs to add (``proposed``)."""

    def setUp(self) -> None:
        products = [
            {"product_code": "100000001", "product_name": "쿠마딘정2밀리그램(와파린나트륨)_(2mg/1정)", "ingredient_code": "W1ATB", "ingredient_name": "warfarin"},
            {"product_code": "100000002", "product_name": "와파린정5밀리그램(와파린나트륨)_(5mg/1정)", "ingredient_code": "W5ATB", "ingredient_name": "warfarin"},
            {"product_code": "200000001", "product_name": "클로시드정250밀리그램(클래리트로마이신)_(0.25g/1정)", "ingredient_code": "C1ATB", "ingredient_name": "clarithromycin"},
            {"product_code": "300000001", "product_name": "조코정20밀리그램(심바스타틴)_(20mg/1정)", "ingredient_code": "S1ATB", "ingredient_name": "simvastatin"},
            # Combination product: its Korean names must not map to its single code.
            {"product_code": "400000001", "product_name": "바이토린정(에제티미브/심바스타틴)_(1정)", "ingredient_code": "E1ATB", "ingredient_name": "ezetimibe + simvastatin"},
        ]
        pair = {"detail": "출혈 위험 증가", "note": "", "notice_no": "1", "notice_date": "2020-01-01"}
        rows = {
            "product": products,
            "pair_rule": [{"ingredient_a": "C1ATB", "ingredient_b": "W1ATB", **pair}, {"ingredient_a": "C1ATB", "ingredient_b": "W5ATB", **pair},
                          {"ingredient_a": "C1ATB", "ingredient_b": "S1ATB", **{**pair, "detail": "근육병증"}}],
            "pregnancy_rule": [{"ingredient_code": "W1ATB", "grade": "1", "detail": "태아 출혈", "notice_no": "2", "notice_date": "2020-01-01"},
                               {"ingredient_code": "W5ATB", "grade": "1", "detail": "태아 출혈", "notice_no": "2", "notice_date": "2020-01-01"}],
        }
        self.checker = DurChecker(DurTables.from_rows({"id": "test"}, rows))

    def test_names_match_korean_without_salt_and_english(self) -> None:
        found = {drug.ingredient_name: drug for drug in self.checker.find_drugs("와파린을 추가하려는데 클래리트로마이신이랑 괜찮아?")}
        self.assertEqual(set(found), {"warfarin", "clarithromycin"})
        self.assertEqual(found["warfarin"].codes, frozenset({"W1ATB", "W5ATB"}))  # every dosage form
        self.assertEqual([drug.ingredient_name for drug in self.checker.find_drugs("Warfarin 시작")], ["warfarin"])
        self.assertEqual(self.checker.find_drugs("혈압 추이 요약"), ())
        self.assertEqual([drug.ingredient_name for drug in self.checker.find_drugs("에제티미브")], [])

    def test_proposed_drug_is_checked_with_prescriptions_once(self) -> None:
        proposed = self.checker.find_drugs("와파린 추가")
        result = self.checker.check([("200000001", "클로시드정")], age_years=30, sex="female", pregnant=True, proposed=proposed).model_dump(mode="json")
        pairs = [alert for alert in result["alerts"] if alert["type"] == "병용금기"]
        self.assertEqual(len(pairs), 1)  # two warfarin dosage forms, one alert
        self.assertEqual([p.get("proposed") for p in pairs[0]["products"]], [None, True])
        pregnancy = [alert for alert in result["alerts"] if alert["type"] == "임부금기"]
        self.assertEqual([(alert["status"], alert["products"][0]["proposed"]) for alert in pregnancy], [("met", True)])
        self.assertEqual(result["proposed"], [{"name": "와파린", "ingredient_name": "warfarin"}])

    def test_a_named_drug_already_prescribed_is_checked_as_the_prescription(self) -> None:
        proposed = self.checker.find_drugs("클래리트로마이신 계속 써도 돼?")
        result = self.checker.check([("200000001", "클로시드정"), ("300000001", "조코정")], age_years=50, sex="male", pregnant=None,
                                    proposed=proposed).model_dump(mode="json")
        self.assertNotIn("proposed", result)
        self.assertEqual([alert["detail"] for alert in result["alerts"]], ["근육병증"])


def _product(code: str, name: str, ingredient_code: str, ingredient_name: str) -> dict[str, str]:
    return {"product_code": code, "product_name": name, "ingredient_code": ingredient_code, "ingredient_name": ingredient_name}


class IngredientIdentityTest(unittest.TestCase):
    """Names are matched to one ingredient: not by the first English word, never to a combination product."""

    def setUp(self) -> None:
        products = [
            _product("1", "빅손정1밀리그람(에칠로플라제페이트)_(1mg/1정)", "156201ATB", "ethyl loflazepate"),
            _product("2", "리피오돌울트라액(아이오다이즈드오일)_(12.8g/10mL)", "156030BIJ", "ethyl esters of the iodised fatty acids of poppyseed oil"),
            _product("3", "포사맥스정70밀리그램(알렌드론산나트륨)_(70mg/1정)", "228301ATB", "sodium alendronate (as alendronic acid)"),
            _product("4", "알렌드로정(알렌드론산나트륨수화물)_(70mg/1정)", "228302ATB", "Alendronate"),
            _product("5", "탄산수소나트륨정(탄산수소나트륨)_(0.5g/1정)", "222001ATB", "sodium bicarbonate"),
            _product("6", "노바스크정5밀리그램(암로디핀베실산염)_(5mg/1정)", "107601ATB", "Amlodipine Besylate (as amlodipine 5mg)"),
            _product("7", "애니디핀정(암로디핀캄실산염)_(5mg/1정)", "459801ATB", "amlodipine camsylate (as amlodipine)"),
            # A combination, as in the list: its products share one code and each names one component.
            _product("8", "엑스포지정5/80밀리그램(암로디핀베실산염)_(1정)", "490001ATB", "amlodipine besylate (as amlodipine)"),
            _product("8b", "엑스포지정10/160밀리그램(암로디핀베실산염)_(1정)", "490001ATB", "valsartan"),
            # A combination whose products all name one component; the rule list names the others.
            _product("18", "콜드정(아세트아미노펜)_(1정)", "C89900ASY", "acetaminophen"),
            # One ingredient spelled two ways on one code is still a single ingredient.
            _product("19", "넥시움정(에스오메프라졸마그네슘)_(20mg/1정)", "367201ATB", "esomeprazole magnesium (as esomeprazol)"),
            _product("20", "에소메졸정(에스오메프라졸마그네슘)_(20mg/1정)", "367201ATB", "esomeprazol magnesium (as esomeprazol)"),
            _product("9", "울트라셋정(트라마돌염산염,아세트아미노펜)_(1정)", "480601ATB", "acetaminophen,tramadol hydrochloride"),
            _product("10", "타이레놀정500밀리그람(아세트아미노펜)_(0.5g/1정)", "101001ATB", "acetaminophen"),
            _product("11", "칼륨정(염화칼륨)_(0.6g/1정)", "211001ATB", "potassium chloride"),
            _product("12", "나트륨주(염화나트륨)_(0.9g/100mL)", "212001BIJ", "sodium chloride"),
            # The Korean name is in an earlier parenthesis than the packaging or export name.
            _product("13", "트리세녹스주(삼산화비소)(바이알)_(12mg/6mL)", "523001BIJ", "arsenic trioxide"),
            _product("14", "클리신정(클로닉신리시네이트)(수출명:CLICIN Tab)_(0.125g/1정)", "135201ATB", "clonixin lysinate"),
            _product("15", "옵디보주100mg(니볼루맙,유전자재조합)_(0.1g/10mL)", "633001BIJ", "nivolumab"),
            # One drug the list spells two ways in English.
            _product("16", "루프린주(류프로렐린아세트산염)_(3.75mg/1병)", "187001BIJ", "leuprorelin"),
            _product("17", "엘리가드주(류프로렐린아세트산염)_(7.5mg/1병)", "187002BIJ", "leuprolide acetate"),
        ]
        rows = {
            "product": products,
            "pregnancy_rule": [
                {"ingredient_code": code, "grade": "1", "detail": detail, "notice_no": "1", "notice_date": "2020-01-01"}
                for code, detail in (("156201ATB", "벤조디아제핀"), ("490001ATB", "ARB 복합제"), ("228301ATB", "비스포스포네이트"))
            ],
            "age_rule": [{"ingredient_code": "C89900ASY", "ingredient_name": "Phenylephrine", "age": "12", "unit": "세", "condition": "미만",
                          "detail": "소아", "notice_no": "1", "notice_date": "2020-01-01"}],
        }
        self.checker = DurChecker(DurTables.from_rows({"id": "test"}, rows))

    def found(self, text: str) -> dict[str, frozenset[str]]:
        return {drug.ingredient_name: drug.codes for drug in self.checker.find_drugs(text)}

    def test_a_shared_first_word_does_not_merge_ingredients(self) -> None:
        self.assertEqual(self.found("아이오다이즈드오일"), {"ethyl esters of the iodised fatty acids of poppyseed oil": {"156030BIJ"}})
        self.assertEqual(self.found("알렌드론산"), {"Alendronate": {"228301ATB", "228302ATB"}})  # both salts, not sodium bicarbonate
        self.assertEqual(self.found("sodium alendronate"), {"Alendronate": {"228301ATB", "228302ATB"}})
        self.assertEqual(self.found("sodium 보충"), {})  # a salt word alone names no ingredient
        self.assertEqual(set(self.found("염화칼륨 염화나트륨")), {"potassium chloride", "sodium chloride"})

    def test_combination_products_are_left_to_prescriptions(self) -> None:
        self.assertEqual(list(self.found("암로디핀").values()), [{"107601ATB", "459801ATB"}])  # both salts, no combination
        self.assertEqual(self.found("amlodipine"), self.found("암로디핀"))
        self.assertEqual(self.found("아세트아미노펜"), {"acetaminophen": {"101001ATB"}})
        self.assertEqual(self.found("트라마돌"), {})
        self.assertEqual(self.found("valsartan"), {})  # only in the combination
        self.assertEqual(self.found("에스오메프라졸"), {"esomeprazole magnesium (as esomeprazol)": {"367201ATB"}})

    def test_korean_name_before_packaging_or_export_name_and_recombinant_biologics(self) -> None:
        self.assertEqual(set(self.found("삼산화비소")), {"arsenic trioxide"})
        self.assertEqual(set(self.found("클로닉신리시네이트")), {"clonixin lysinate"})
        self.assertEqual(set(self.found("니볼루맙")), {"nivolumab"})
        self.assertEqual(self.found("바이알 주사"), {})

    def test_one_drug_spelled_two_ways_is_one_drug(self) -> None:
        (drug,) = self.checker.find_drugs("류프로렐린 주사")
        self.assertEqual(drug.codes, {"187001BIJ", "187002BIJ"})

    def test_prescribed_ingredient_is_not_rechecked_but_one_sharing_a_word_is(self) -> None:
        # Sodium bicarbonate prescribed; alendronate named: still checked (before, both were "sodium" and it was skipped).
        result = self.checker.check([("5", None)], age_years=30, sex="female", pregnant=True,
                                    proposed=self.checker.find_drugs("알렌드론산 추가")).model_dump(mode="json")
        self.assertEqual([alert["detail"] for alert in result["alerts"]], ["비스포스포네이트"])
        self.assertEqual(result["proposed"], [{"name": "알렌드론산", "ingredient_name": "Alendronate"}])
        # Amlodipine is in the prescribed combination: checked once, as the prescription (the combination's own rule).
        result = self.checker.check([("8", None)], age_years=30, sex="female", pregnant=True,
                                    proposed=self.checker.find_drugs("암로디핀 추가")).model_dump(mode="json")
        self.assertNotIn("proposed", result)
        self.assertEqual([alert["detail"] for alert in result["alerts"]], ["ARB 복합제"])

    def test_named_amlodipine_alone_gets_no_combination_rule(self) -> None:
        result = self.checker.check([], age_years=30, sex="female", pregnant=True, proposed=self.checker.find_drugs("amlodipine")).model_dump(mode="json")
        self.assertEqual(result["alerts"], [])


class RepositoryNameMatchingTest(unittest.TestCase):
    """The reported cases, on the DUR snapshot in the repository."""

    @classmethod
    def setUpClass(cls) -> None:
        snapshot = read_snapshot(REPO_SNAPSHOT)
        cls.checker = DurChecker(DurTables.from_rows(snapshot.source, snapshot.rows))
        names: dict[str, set[str]] = {}
        for row in snapshot.rows["product"]:
            names.setdefault(row["ingredient_code"], set()).add(row["ingredient_name"].lower())
        cls.names = names

    def ingredients(self, text: str) -> set[str]:
        return {name for drug in self.checker.find_drugs(text) for code in drug.codes for name in self.names[code]}

    def test_reported_cases(self) -> None:
        self.assertNotIn("ethyl loflazepate", self.ingredients("아이오다이즈드오일"))
        self.assertFalse(any("bicarbonate" in name or "phytate" in name for name in self.ingredients("알렌드론산")))
        self.assertTrue(all("amlodipine" in name for name in self.ingredients("amlodipine")))  # no valsartan, no combination
        self.assertFalse(any("tramadol" in name for name in self.ingredients("acetaminophen")))
        self.assertEqual(self.checker.find_drugs("sodium"), ())


if __name__ == "__main__":
    unittest.main()
