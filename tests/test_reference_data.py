from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ax_g_ai.reference_data import ReferenceDataError, convert_reference_pickles


class ReferenceDataConversionTest(unittest.TestCase):
    def test_converts_legacy_dictionaries_and_preserves_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            medicine = root / "medicine.pkl"
            disease = root / "disease.pkl"
            output = root / "reference.sqlite"
            medicine.write_bytes(pickle.dumps({
                "100": {
                    "name_ko": "검사용 약품",
                    "efficacy": "검사용 효능",
                    "usagex": "검사용 용법",
                    "carefully": float("nan"),
                }
            }, protocol=4))
            disease.write_bytes(pickle.dumps({"A000": "검사용 질환"}, protocol=4))

            self.assertEqual(
                convert_reference_pickles(
                    medicine_pickle=medicine, disease_pickle=disease, output=output
                ),
                (1, 1),
            )

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    connection.execute("SELECT code, name_ko FROM diseases").fetchall(),
                    [("A000", "검사용 질환")],
                )
                self.assertEqual(
                    connection.execute(
                        """SELECT source_kd_code, kd_code, name_ko, efficacy, usage, precautions
                        FROM medicines"""
                    ).fetchall(),
                    [("100", "100", "검사용 약품", "검사용 효능", "검사용 용법", None)],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key = 'schema_version'"
                    ).fetchone(),
                    ("1",),
                )

    def test_rejects_pickle_with_object_construction_opcode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            medicine = root / "unsafe.pkl"
            disease = root / "disease.pkl"
            medicine.write_bytes(pickle.dumps({"item": Path("unsafe")}, protocol=4))
            disease.write_bytes(pickle.dumps({"A000": "검사용 질환"}, protocol=4))

            with self.assertRaises(ReferenceDataError):
                convert_reference_pickles(
                    medicine_pickle=medicine, disease_pickle=disease, output=root / "reference.sqlite"
                )


if __name__ == "__main__":
    unittest.main()
