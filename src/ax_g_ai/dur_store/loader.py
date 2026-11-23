"""DUR 스냅샷(``data/dur``, ``snapshot`` 참고)을 PostgreSQL ``dur`` 스키마에 데이터셋으로 적재하는 모듈과 CLI.

설계상 위치: DUR 저장 계층의 쓰기 경로. compose 일회성 서비스 ``dur-loader``(``ax-g-ai-load-dur``)가 실행하며,
서비스 런타임은 결과를 ``reader``로 읽는다.

역할: 스냅샷 검증 → 스키마 보장 → 같은 내용 해시의 데이터셋 재사용 또는 새 비활성 데이터셋 삽입(COPY) → 행 수 검증 →
원자적 활성화 → 오래된 데이터셋 정리. 런타임은 ``reader``로 활성 데이터셋을 읽는다(시작 시, 그리고 활성
데이터셋이 바뀌면 ``corpus_store.live.LiveCorpus``가 다시 읽는다).

CLI(compose 일회성 서비스 ``dur-loader``)::

    ax-g-ai-load-dur                       # 환경 변수: AX_G_CORPUS_DATABASE_URL, AX_G_DUR_SNAPSHOT_DIR
    ax-g-ai-load-dur --activate 2          # 데이터셋 2로 되돌리기
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ax_g_ai.config import ConfigurationError, LoaderSettings
from ax_g_ai.dur_store.snapshot import TABLE_COLUMNS, DurDataError, Snapshot, read_snapshot
from ax_g_ai.pg_release import ReleaseTable, scalar

logger = logging.getLogger("ax_g_ai.dur_store")

KEEP_DATASETS = 3


class DurLoadError(RuntimeError):
    """적재한 데이터셋이 검증에 실패했거나 요청한 데이터셋이 없을 때의 오류.

    역할: CLI가 잡아 종료 코드 1로 보고하는 적재 실패 예외(``ReleaseTable.activate``도 이 예외를 쓴다).
    """


@dataclass
class DurLoadResult:
    """DUR 적재 한 번의 결과.

    역할: 데이터셋 ID, 재사용 여부, 표별 행 수를 CLI 출력용으로 담는다.
    """

    dataset_id: int
    reused: bool
    counts: dict[str, int]


DATASETS = ReleaseTable("dur", "dataset", "dataset_id", "ax_g_ai.dur_store", KEEP_DATASETS, DurLoadError, "dataset")
ensure_schema = DATASETS.ensure_schema
activate = DATASETS.activate  # 롤백도 이것으로 한다: 이전 데이터셋을 활성화


def load_dur(conn: psycopg.Connection, snapshot_dir: Path) -> DurLoadResult:
    """스냅샷 폴더를 데이터셋으로 적재하고 활성화한다.

    역할: 스냅샷을 검증해 읽고, 같은 ``content_sha256``의 데이터셋이 있으면 재사용, 없으면 한 트랜잭션에서 새로 넣는다.
    이어 행 수를 검증하고 활성화한 뒤 오래된 데이터셋을 정리한다.
    입력:
        conn: autocommit PostgreSQL 연결.
        snapshot_dir: 스냅샷 폴더.
    출력: ``DurLoadResult``.
    예외: 스냅샷 오류면 ``DurDataError``, 검증 실패면 ``DurLoadError``, DB 오류면 ``psycopg.Error``.
    """
    snapshot = read_snapshot(snapshot_dir)
    ensure_schema(conn)
    counts = {name: len(rows) for name, rows in snapshot.rows.items()}
    existing = conn.execute(
        "SELECT dataset_id FROM dur.dataset WHERE content_sha256 = %s ORDER BY dataset_id DESC LIMIT 1",
        (snapshot.content_sha256,),
    ).fetchone()
    if existing is not None:
        dataset_id, reused = existing[0], True
    else:
        with conn.transaction():
            dataset_id = _insert_dataset(conn, snapshot)
        reused = False
    _validate(conn, dataset_id, counts)
    activate(conn, dataset_id)
    DATASETS.prune(conn)
    return DurLoadResult(dataset_id, reused, counts)


def _insert_dataset(conn: psycopg.Connection, snapshot: Snapshot) -> int:
    """데이터셋 행을 만들고 표마다 COPY로 행을 넣는다.

    역할: manifest의 출처·빌드 시각·입력 파일·행 수·내용 해시를 ``dur.dataset``에 기록하고,
    ``TABLE_COLUMNS`` 순서로 각 ``dur.<표>``에 행을 복사한다.
    입력:
        conn: PostgreSQL 연결(호출자가 트랜잭션을 연다).
        snapshot: 검증된 스냅샷.
    출력: 새 데이터셋 ID.
    """
    source, manifest = snapshot.source, snapshot.manifest
    dataset_id = scalar(conn.execute(
        "INSERT INTO dur.dataset (source_id, title, url, license, region, built_at, inputs, counts, content_sha256) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING dataset_id",
        (source["id"], source["title"], source.get("url"), source["license"], source["region"], manifest["built_at"],
         Jsonb(manifest.get("inputs") or {}), Jsonb({name: len(rows) for name, rows in snapshot.rows.items()}),
         snapshot.content_sha256),
    ))
    with conn.cursor() as cur:
        for table, columns in TABLE_COLUMNS.items():
            with cur.copy(f"COPY dur.{table} (dataset_id, {', '.join(columns)}) FROM STDIN") as copy:
                for row in snapshot.rows[table]:
                    copy.write_row((dataset_id, *(row[column] for column in columns)))
    return dataset_id


def _validate(conn: psycopg.Connection, dataset_id: int, counts: dict[str, int]) -> None:
    """표마다 DB 행 수가 스냅샷 행 수와 같은지 확인한다. 다르면 ``DurLoadError``."""
    for table, expected in counts.items():
        actual = scalar(conn.execute(f"SELECT count(*) FROM dur.{table} WHERE dataset_id = %s", (dataset_id,)))
        if actual != expected:
            raise DurLoadError(f"dataset {dataset_id}: {table} has {actual} rows, expected {expected}")


def main(argv: list[str] | None = None) -> int:
    """``ax-g-ai-load-dur`` CLI 진입점.

    역할: 환경 설정을 읽고, ``--activate``면 해당 데이터셋만 활성화(롤백)하고, 아니면 스냅샷을 적재해 결과를 출력한다.
    오류 출력에는 오류 클래스와 메시지만 쓰며 DB URL은 출력하지 않는다.
    입력:
        argv: 명령행 인자(``None``이면 ``sys.argv``).
    출력: 종료 코드. 성공 0, 적재 실패 1, 설정 오류 2.
    """
    parser = argparse.ArgumentParser(description="Load the HIRA DUR snapshot into PostgreSQL (schema dur).")
    parser.add_argument("--snapshot-dir", type=Path, help="snapshot folder (default: AX_G_DUR_SNAPSHOT_DIR or data/dur)")
    parser.add_argument("--activate", type=int, metavar="DATASET_ID", help="activate an existing dataset (rollback) and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = LoaderSettings.from_environment()
    except ConfigurationError as error:  # 필드 이름과 사유만 담고 값은 절대 담지 않는다
        print(f"invalid configuration: {error}", file=sys.stderr)
        return 2
    try:
        with psycopg.connect(settings.corpus_database_url.get_secret_value(), autocommit=True, connect_timeout=10) as conn:
            if args.activate is not None:
                ensure_schema(conn)
                activate(conn, args.activate)
                print(f"activated dur dataset {args.activate}")
                return 0
            result = load_dur(conn, args.snapshot_dir or settings.dur_snapshot_dir)
    except (psycopg.Error, DurLoadError, DurDataError, OSError) as error:
        # 오류 클래스와 메시지만 출력한다. (자격 증명이 담긴) DB URL은 절대 출력하지 않는다.
        print(f"dur load failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"dur dataset {result.dataset_id} active ({'reused' if result.reused else 'new'}): "
          + ", ".join(f"{name} {count}" for name, count in result.counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
