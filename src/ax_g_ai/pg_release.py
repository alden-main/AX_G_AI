"""PostgreSQL의 버전별 데이터셋 관리: 활성 행 하나, 원자적 전환, 이력 개수 제한.

설계상 위치: 저장소(store) 계층의 공용 도구. 말뭉치(``corpus.corpus_release``)와 DUR(``dur.dataset``)
적재기·읽기 모듈이 ``ReleaseTable``과 ``scalar``·``migrations``를 쓴다.

역할: 적재기는 적재할 때마다 새 행을 저장하고 한 트랜잭션에서 ``is_active``를 전환한다. 롤백은 예전 행을
다시 활성화하는 것이다.

스키마 변경은 각 저장소 패키지 안의 번호 붙은 SQL 마이그레이션(``migrations/NNNN_name.sql``)이다.
``ensure_schema``는 ``<schema>.schema_version``에 아직 기록되지 않은 마이그레이션을 번호 순서대로, advisory lock을 잡은
한 트랜잭션에서 적용한다. 그래서 적재기 두 개가 동시에 시작해도 같은 마이그레이션을 두 번 적용하지 않고, 실패한
마이그레이션은 스키마를 원래대로 남긴다. 마이그레이션이 생기기 전에 만든 DB에는 버전 1(``0001_initial.sql``)이
기록되어 있다. 따라서 마이그레이션은 트랜잭션 안에서 유효해야 하며(``CREATE INDEX CONCURRENTLY`` 불가), 이미 적용한
마이그레이션은 고치지 않고 새 마이그레이션을 추가한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row

_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def scalar(cursor: psycopg.Cursor[Any]) -> Any:
    """문장이 돌려준 한 행의 첫 열(``count(*)``, ``RETURNING id``)을 꺼낸다. 행이 없으면 ``psycopg.DataError``다."""
    row = cursor.fetchone()
    if row is None:
        raise psycopg.DataError("statement returned no row")
    return row[0] if isinstance(row, (tuple, list)) else next(iter(row.values()))


class MigrationError(RuntimeError):
    """패키지에 든 마이그레이션 파일이 잘못되었다(버전 번호 중복 또는 해석 불가).

    역할: ``migrations``가 파일 이름 규칙 위반·버전 중복을 알릴 때 발생시킨다.
    """


def migrations(package: str) -> list[tuple[int, str, str]]:
    """``package``/migrations에 든 모든 마이그레이션을 버전 순서로 돌려준다.

    역할: ``.sql`` 파일만 보며, 이름이 ``NNNN_name.sql`` 규칙을 따르는지와 버전 중복을 검사한다.
    입력:
        package: 마이그레이션 폴더를 가진 패키지 이름(예: ``ax_g_ai.corpus_store``).
    출력: (버전, 파일 이름, SQL 본문) 튜플 목록. 버전 오름차순.
    예외: ``MigrationError`` — 이름 규칙 위반 또는 같은 버전이 둘 이상.
    """
    found: dict[int, tuple[int, str, str]] = {}
    for entry in resources.files(package).joinpath("migrations").iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = _MIGRATION_NAME.match(entry.name)
        if match is None:
            raise MigrationError(f"{package}: migration file name must be NNNN_name.sql: {entry.name}")
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"{package}: two migrations with version {version}")
        found[version] = (version, entry.name, entry.read_text(encoding="utf-8"))
    return [found[version] for version in sorted(found)]


@dataclass(frozen=True)
class ReleaseTable:
    """버전별 데이터셋 테이블 하나(말뭉치 릴리스 또는 DUR 데이터셋)의 설정과 관리 연산.

    역할: 스키마 마이그레이션 적용(``ensure_schema``), 지정 행 활성화(``activate``), 오래된 비활성 행
    정리(``prune``)를 테이블 이름·ID 열만 바꿔 두 저장소가 함께 쓰게 한다.
    """

    schema: str              # PostgreSQL 스키마(예: "corpus")
    table: str               # 릴리스 테이블(예: "corpus_release")
    id_column: str           # 예: "release_id"
    schema_package: str      # ``migrations/NNNN_name.sql``을 가진 패키지
    keep: int                # 남겨 둘 최신 행 개수(활성 행은 이와 관계없이 남는다)
    error: type[Exception]   # 없는 릴리스 ID일 때 발생시킬 예외
    label: str               # 메시지용 이름("release" / "dataset")

    @property
    def _table(self) -> sql.Composable:
        return sql.Identifier(self.schema, self.table)

    def ensure_schema(self, conn: psycopg.Connection) -> list[int]:
        """아직 적용하지 않은 마이그레이션을 적용한다.

        역할: advisory lock을 잡은 한 트랜잭션에서 스키마와 ``schema_version`` 테이블을 만들고(없으면),
        기록되지 않은 마이그레이션을 버전 순서로 실행·기록한다. 스크립트마다 실행 후 ``search_path``를 기본값으로 되돌린다.
        입력:
            conn: PostgreSQL 연결.
        출력: 이번에 적용한 버전 목록(스키마가 최신이면 빈 목록).
        """
        schema = sql.Identifier(self.schema)
        versions = sql.Identifier(self.schema, "schema_version")
        pending: list[int] = []
        with conn.transaction():
            cursor = conn.cursor(row_factory=tuple_row)
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"ax_g_ai.migrate.{self.schema}",))
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
            cursor.execute(sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            ).format(versions))
            applied = {row[0] for row in cursor.execute(sql.SQL("SELECT version FROM {}").format(versions)).fetchall()}
            for version, _name, script in migrations(self.schema_package):
                if version in applied:
                    continue
                cursor.execute(script)  # type: ignore[arg-type]  # 패키지에 함께 배포한 신뢰된 파일
                cursor.execute("SET search_path TO DEFAULT")  # 스크립트는 먼저 자기 스키마를 지정한다
                cursor.execute(sql.SQL("INSERT INTO {} (version) VALUES (%s)").format(versions), (version,))
                pending.append(version)
        return pending

    def activate(self, conn: psycopg.Connection, release_id: int) -> None:
        """지정한 행을 활성 데이터셋으로 전환한다.

        역할: 한 트랜잭션에서 다른 활성 행을 끄고 ``release_id`` 행만 켠다. 예전 행을 지정하면 롤백이 된다.
        입력:
            conn: PostgreSQL 연결.
            release_id: 활성화할 행의 ID.
        출력: 없음.
        예외: 없는 ID면 ``error``에 지정한 예외.
        """
        column = sql.Identifier(self.id_column)
        with conn.transaction():
            exists = conn.execute(sql.SQL("SELECT 1 FROM {} WHERE {} = %s").format(self._table, column), (release_id,))
            if exists.fetchone() is None:
                raise self.error(f"{self.label} {release_id} does not exist")
            conn.execute(sql.SQL("UPDATE {} SET is_active = false WHERE is_active AND {} <> %s").format(self._table, column),
                         (release_id,))
            conn.execute(sql.SQL("UPDATE {} SET is_active = true WHERE {} = %s").format(self._table, column), (release_id,))

    def prune(self, conn: psycopg.Connection) -> None:
        """ID 기준 최신 ``keep``개 밖의 비활성 행을 지운다(하위 행은 cascade로 함께 지워진다)."""
        column = sql.Identifier(self.id_column)
        with conn.transaction():
            conn.execute(
                sql.SQL("DELETE FROM {table} WHERE NOT is_active AND {id} NOT IN ("
                        "SELECT {id} FROM {table} ORDER BY {id} DESC LIMIT %s)").format(table=self._table, id=column),
                (self.keep,),
            )
