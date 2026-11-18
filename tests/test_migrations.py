from __future__ import annotations

import unittest
from unittest import mock

from ax_g_ai import pg_release


class PackagedMigrationsTest(unittest.TestCase):
    def test_each_store_ships_numbered_migrations_starting_at_the_initial_schema(self) -> None:
        for package in ("ax_g_ai.corpus_store", "ax_g_ai.dur_store"):
            with self.subTest(package=package):
                found = pg_release.migrations(package)
                self.assertEqual(found[0][:2], (1, "0001_initial.sql"))
                self.assertEqual([version for version, _name, _sql in found], list(range(1, len(found) + 1)))
                # The runner records versions; a script must not create the table itself.
                self.assertFalse(any("CREATE TABLE schema_version" in script for _v, _n, script in found))

    def test_malformed_names_and_duplicate_versions_are_rejected(self) -> None:
        class Entry:
            def __init__(self, name: str) -> None:
                self.name = name

            def read_text(self, encoding: str) -> str:
                return ""

        for names in (["0001_a.sql", "0001_b.sql"], ["1_initial.sql"], ["0002_Bad-Name.sql"]):
            with self.subTest(names=names):
                folder = mock.Mock(iterdir=lambda names=names: [Entry(name) for name in names])
                root = mock.Mock(joinpath=lambda _name, folder=folder: folder)
                with mock.patch.object(pg_release.resources, "files", return_value=root), self.assertRaises(pg_release.MigrationError):
                    pg_release.migrations("pkg")


if __name__ == "__main__":
    unittest.main()
