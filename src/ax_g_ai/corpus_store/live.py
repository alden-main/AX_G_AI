"""서빙 중인 말뭉치 스냅샷을 PostgreSQL의 활성 릴리스·활성 DUR 데이터셋과 맞춰 두는 모듈.

설계상 위치: 말뭉치 저장 계층과 서비스 계층의 접점. 앱 시작 시 만들어지고, 요청 처리 코드는 ``LiveCorpus.current``로
현재 ``CorpusSnapshot``을 얻는다.

역할: 백그라운드 스레드 하나가 ``refresh_seconds``마다 두 ID(말뭉치 릴리스, DUR 데이터셋)를 확인한다(작은 쿼리 두 개).
둘 중 하나라도 바뀌면 완전한 새 스냅샷(근거 색인, 규칙, DUR)을 만들어 대입 한 번으로 교체한다. 요청은 재적재를
기다리지 않고, 반쯤 바뀐 상태도 보지 않는다. 빌드가 실패하면 현재 스냅샷을 유지하고 다음 확인 때 다시 시도한다.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence

import psycopg
from psycopg_pool import ConnectionPool

from ax_g_ai.corpus_store.retriever import active_release_id
from ax_g_ai.dur_store.reader import active_dataset_id
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot
from ax_g_ai.services.dur_check import DurUnavailableError

logger = logging.getLogger("uvicorn.error")

# (연결, 말뭉치 릴리스 ID, DUR 데이터셋 ID 또는 None) → 그 릴리스·데이터셋의 스냅샷
SnapshotBuilder = Callable[[psycopg.Connection, int, "int | None"], CorpusSnapshot]


class LiveCorpus:
    """활성 릴리스가 바뀌면 스냅샷을 원자적으로 교체하는 말뭉치 보관자.

    역할: 생성 시 첫 스냅샷을 만들고 갱신 스레드를 띄운다. ``current``는 항상 완전한 스냅샷을 돌려주며,
    ``refresh``·``close``·``database_reachable``로 교체·종료·DB 상태 확인을 제공한다.
    """

    def __init__(
        self, pool: ConnectionPool, build: SnapshotBuilder, *, refresh_seconds: float = 60.0, watch_dur: bool = True,
        on_close: Sequence[Callable[[], None]] = (),
    ) -> None:
        """첫 스냅샷을 만들고 백그라운드 갱신 스레드를 시작한다.

        역할: 연결 풀·빌더·설정을 보관하고 ``_load``로 초기 스냅샷을 만든 뒤 데몬 스레드 ``corpus-refresh``를 띄운다.
        입력:
            pool: 공유 PostgreSQL 연결 풀(``close`` 때 함께 닫는다).
            build: (연결, 릴리스 ID, DUR 데이터셋 ID) → ``CorpusSnapshot`` 빌더.
            refresh_seconds: 활성 ID 확인 주기(초).
            watch_dur: True면 활성 DUR 데이터셋 ID도 확인한다. False면 DUR ID는 항상 ``None``.
            on_close: ``close`` 때 차례로 부를 정리 함수들(예: 임베딩 클라이언트 닫기).
        출력: 없음.
        예외: 시작 중 활성 릴리스가 바뀌어 첫 빌드를 버리면 ``psycopg.OperationalError``(재시도 요청).
            DB 오류·활성 릴리스 없음은 그대로 전파된다.
        """
        self._pool = pool
        self._build = build
        self._watch_dur = watch_dur
        self._on_close = tuple(on_close)
        self._refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        snapshot = self._load()
        if snapshot is None:
            raise psycopg.OperationalError("active corpus release changed during start-up; retry")
        self._current = snapshot
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="corpus-refresh", daemon=True)
        self._thread.start()

    @property
    def current(self) -> CorpusSnapshot:
        """지금 서빙 중인 스냅샷."""
        return self._current

    def database_reachable(self) -> bool:
        """풀을 통해 ``SELECT 1``을 한 번 실행해 DB 연결 가능 여부를 돌려준다(풀의 3초 제한 적용).

        역할: 준비 상태 확인용. 결과와 관계없이 스냅샷은 계속 서빙된다. DB가 내려가면 의미 검색과 릴리스 확인만
        DB가 돌아올 때까지 꺼진다.
        입력: 없음.
        출력: 연결되면 True, ``psycopg.Error``가 나면 경고 로그를 남기고 False.
        """
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
        except psycopg.Error as error:  # PoolTimeout은 psycopg.OperationalError의 하위 클래스
            logger.warning("corpus_database_ping_failed error_type=%s", type(error).__name__)
            return False
        return True

    def refresh(self) -> bool:
        """활성 릴리스나 데이터셋이 바뀌었으면 새 스냅샷으로 교체한다.

        역할: 잠금 안에서 현재 활성 ID를 현재 스냅샷의 ID와 비교하고, 다르면 ``_load``로 새로 만들어 대입한다.
        교체하면 이전·새 ID를 로그로 남긴다.
        입력: 없음.
        출력: 교체했으면 True. ID가 같거나 빌드 중 릴리스가 또 바뀌어 버렸으면 False.
        """
        with self._lock:
            with self._pool.connection() as conn:
                ids = self._ids(conn)
            current = self._current
            if ids == (current.release_id, current.dur_dataset_id):
                return False
            snapshot = self._load()
            if snapshot is None:
                return False
            self._current = snapshot
        logger.info(
            "corpus_snapshot_swapped release_id=%s dur_dataset_id=%s previous_release_id=%s previous_dur_dataset_id=%s",
            snapshot.release_id, snapshot.dur_dataset_id, current.release_id, current.dur_dataset_id,
        )
        return True

    def close(self) -> None:
        """갱신 스레드를 멈추고(최대 5초 대기) 풀을 닫은 뒤 ``on_close`` 함수들을 부른다."""
        self._stop.set()
        self._thread.join(timeout=5)
        self._pool.close()
        for close in self._on_close:
            close()

    def _ids(self, conn: psycopg.Connection) -> tuple[int, int | None]:
        """(활성 말뭉치 릴리스 ID, 활성 DUR 데이터셋 ID 또는 None).

        역할: DUR 감시가 꺼져 있거나 DUR 데이터를 쓸 수 없으면(``DurUnavailableError``) DUR ID를 ``None``으로 둔다.
        입력:
            conn: PostgreSQL 연결.
        출력: 두 ID의 튜플.
        예외: 활성 말뭉치 릴리스가 없으면 ``CorpusUnavailableError``.
        """
        dataset_id = None
        if self._watch_dur:
            try:
                dataset_id = active_dataset_id(conn)
            except DurUnavailableError:
                dataset_id = None
        return active_release_id(conn), dataset_id

    def _load(self) -> CorpusSnapshot | None:
        """현재 활성 ID로 스냅샷을 만들고, 빌드 전후 ID가 같을 때만 돌려준다.

        역할: 빌드 도중 릴리스가 전환되면 게이트 뷰가 다른 릴리스를 보여 주므로 결과가 섞였을 수 있어 버린다.
        입력: 없음.
        출력: 새 ``CorpusSnapshot``. 빌드 중 ID가 바뀌었으면 None.
        """
        with self._pool.connection() as conn:
            ids = self._ids(conn)
            snapshot = self._build(conn, *ids)
            if self._ids(conn) != ids:
                # 빌드 중 전환됨: 게이트 뷰가 이제 다른 릴리스를 보여 주므로 이 빌드는 일부만 반영됐을 수 있다.
                logger.warning("corpus_snapshot_discarded reason=release_switched_during_build")
                return None
        return snapshot

    def _run(self) -> None:
        """중지 신호가 올 때까지 ``refresh_seconds``마다 ``refresh``를 부르는 스레드 본문(실패는 경고로만 남긴다)."""
        while not self._stop.wait(self._refresh_seconds):
            try:
                self.refresh()
            except Exception as error:  # 현재 스냅샷을 계속 서빙한다
                logger.warning("corpus_refresh_failed error_type=%s", type(error).__name__)
