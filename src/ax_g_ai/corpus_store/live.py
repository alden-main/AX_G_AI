"""Keep the served corpus snapshot on the active PostgreSQL release and DUR dataset.

One background thread checks both ids every ``refresh_seconds`` (two small queries). When either
changed, it builds a complete new snapshot (evidence index, rules, DUR) and swaps it in with one
assignment; requests never wait for a reload and never see half of one. A failed build keeps the
current snapshot and is retried on the next check.
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

# (connection, corpus release id, DUR dataset id or None) → snapshot of that release and dataset
SnapshotBuilder = Callable[[psycopg.Connection, int, "int | None"], CorpusSnapshot]


class LiveCorpus:
    def __init__(
        self, pool: ConnectionPool, build: SnapshotBuilder, *, refresh_seconds: float = 60.0, watch_dur: bool = True,
        on_close: Sequence[Callable[[], None]] = (),
    ) -> None:
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
        return self._current

    def refresh(self) -> bool:
        """Swap in a new snapshot if the active release or dataset changed; True when swapped."""
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
        self._stop.set()
        self._thread.join(timeout=5)
        self._pool.close()
        for close in self._on_close:
            close()

    def _ids(self, conn: psycopg.Connection) -> tuple[int, int | None]:
        dataset_id = None
        if self._watch_dur:
            try:
                dataset_id = active_dataset_id(conn)
            except DurUnavailableError:
                dataset_id = None
        return active_release_id(conn), dataset_id

    def _load(self) -> CorpusSnapshot | None:
        with self._pool.connection() as conn:
            ids = self._ids(conn)
            snapshot = self._build(conn, *ids)
            if self._ids(conn) != ids:
                # Switched while building: the gate view now serves another release, so this build may be partial.
                logger.warning("corpus_snapshot_discarded reason=release_switched_during_build")
                return None
        return snapshot

    def _run(self) -> None:
        while not self._stop.wait(self._refresh_seconds):
            try:
                self.refresh()
            except Exception as error:  # keep serving the current snapshot
                logger.warning("corpus_refresh_failed error_type=%s", type(error).__name__)
