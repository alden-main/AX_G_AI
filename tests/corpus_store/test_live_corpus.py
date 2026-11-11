"""LiveCorpus swaps the whole snapshot when the active corpus release or DUR dataset changes."""

from __future__ import annotations

import contextlib
import time
import unittest
from unittest.mock import patch

from ax_g_ai.corpus_store.live import LiveCorpus
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot
from ax_g_ai.services.dur_check import DurUnavailableError


class FakePool:
    closed = False

    def connection(self):
        return contextlib.nullcontext(object())

    def close(self) -> None:
        self.closed = True


class LiveCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids: dict[str, object] = {"release": 1, "dur": 10}

        def dataset_id(_conn):
            if self.ids["dur"] is None:
                raise DurUnavailableError("no active dur dataset")
            return self.ids["dur"]

        for target, value in (("active_release_id", lambda _conn: self.ids["release"]), ("active_dataset_id", dataset_id)):
            patcher = patch(f"ax_g_ai.corpus_store.live.{target}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.builds: list[tuple[object, object]] = []
        self.pool = FakePool()

    def build(self, _conn, release_id, dataset_id) -> CorpusSnapshot:
        self.builds.append((release_id, dataset_id))
        return CorpusSnapshot(release_id=release_id, dur_dataset_id=dataset_id)

    def live(self, build=None, **options) -> LiveCorpus:
        corpus = LiveCorpus(self.pool, build or self.build, refresh_seconds=options.pop("refresh_seconds", 3600), **options)  # type: ignore[arg-type]
        self.addCleanup(corpus.close)
        return corpus

    def test_unchanged_ids_keep_the_snapshot(self) -> None:
        corpus = self.live()
        first = corpus.current
        self.assertFalse(corpus.refresh())
        self.assertIs(corpus.current, first)
        self.assertEqual(self.builds, [(1, 10)])

    def test_a_new_release_or_dataset_swaps_everything_at_once(self) -> None:
        corpus = self.live()
        self.ids["release"] = 2
        self.assertTrue(corpus.refresh())
        self.assertEqual((corpus.current.release_id, corpus.current.dur_dataset_id), (2, 10))
        self.ids["dur"] = 11
        self.assertTrue(corpus.refresh())
        self.assertEqual((corpus.current.release_id, corpus.current.dur_dataset_id), (2, 11))
        self.ids["dur"] = None  # DUR dataset removed: rules stay, DUR check is off
        self.assertTrue(corpus.refresh())
        self.assertIsNone(corpus.current.dur_dataset_id)

    def test_failed_build_keeps_the_current_snapshot(self) -> None:
        corpus = self.live()
        first = corpus.current
        self.ids["release"] = 2

        def failing(*_args):
            raise RuntimeError("bad release")

        corpus._build = failing  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):  # the refresh thread logs it and retries on the next check
            corpus.refresh()
        self.assertIs(corpus.current, first)

    def test_release_switched_during_build_is_discarded(self) -> None:
        def switching(conn, release_id, dataset_id):
            snapshot = self.build(conn, release_id, dataset_id)
            self.ids["release"] = 3  # activated by a loader while we were reading release 2
            return snapshot

        corpus = self.live()
        self.ids["release"] = 2
        corpus._build = switching  # type: ignore[assignment]
        with self.assertLogs("uvicorn.error", level="WARNING"):
            self.assertFalse(corpus.refresh())
        self.assertEqual(corpus.current.release_id, 1)
        corpus._build = self.build  # type: ignore[assignment]
        self.assertTrue(corpus.refresh())
        self.assertEqual(corpus.current.release_id, 3)

    def test_background_thread_picks_up_a_new_release_and_close_stops_it(self) -> None:
        closed: list[str] = []
        corpus = LiveCorpus(self.pool, self.build, refresh_seconds=0.02, on_close=(lambda: closed.append("embedder"),))  # type: ignore[arg-type]
        self.ids["release"] = 5
        deadline = time.monotonic() + 2
        while corpus.current.release_id != 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(corpus.current.release_id, 5)
        corpus.close()
        self.assertTrue(self.pool.closed)
        self.assertEqual(closed, ["embedder"])
        self.assertFalse(corpus._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
