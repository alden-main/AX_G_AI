"""Import rules between packages, so the layering does not drift back.

    corpus, jsonl  (standard library only: the corpus contract)
        ↑
    services       (domain and use cases; defines the provider port in services.ports)
        ↑
    adapters, corpus_store, dur_store   (implement ports, read/write external systems)
        ↑
    api, runtime   (HTTP edge and composition)

The offline loaders and build checks (corpus/DUR loaders, DUR snapshot, precomputed check) also
stay off ``services`` and ``adapters``, directly or indirectly, so they never import request-path code.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ax_g_ai"


def imported_modules(path: Path) -> set[str]:
    """Every ``ax_g_ai`` module the file imports, at any depth (lazy imports included)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level:
            package = ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts[: -node.level])
            found.add(f"{package}.{node.module}" if node.module else package)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name for name in found if name == "ax_g_ai" or name.startswith("ax_g_ai.")}


def module_files(*packages: str) -> list[Path]:
    files: list[Path] = []
    for package in packages:
        target = PACKAGE / package
        files += sorted(target.rglob("*.py")) if target.is_dir() else [target.with_suffix(".py")]
    return files


class LayeringTest(unittest.TestCase):
    def assert_no_imports(self, files: list[Path], forbidden: tuple[str, ...], allowed: tuple[str, ...] = ()) -> None:
        self.assertTrue(files)
        for path in files:
            for module in imported_modules(path):
                if module.startswith(allowed):
                    continue
                with self.subTest(file=str(path.relative_to(PACKAGE)), imports=module):
                    self.assertFalse(module.startswith(forbidden), f"{path.relative_to(PACKAGE)} imports {module}")

    def test_corpus_contract_depends_on_nothing_in_the_project(self) -> None:
        self.assert_no_imports(module_files("corpus", "jsonl"), ("ax_g_ai",), allowed=("ax_g_ai.corpus",))

    def test_services_do_not_import_adapters_stores_or_the_edge(self) -> None:
        self.assert_no_imports(
            module_files("services", "prompts"),
            ("ax_g_ai.adapters", "ax_g_ai.corpus_store", "ax_g_ai.dur_store", "ax_g_ai.api", "ax_g_ai.runtime", "ax_g_ai.config"),
        )

    def test_adapters_and_stores_do_not_import_the_edge(self) -> None:
        self.assert_no_imports(module_files("adapters", "corpus_store", "dur_store"), ("ax_g_ai.api", "ax_g_ai.runtime"))

    def test_offline_loaders_do_not_import_request_path_services(self) -> None:
        # Checked in a fresh interpreter so indirect imports count too.
        modules = ["ax_g_ai.corpus_store.loader", "ax_g_ai.corpus_store.precomputed", "ax_g_ai.dur_store.loader",
                   "ax_g_ai.dur_store.snapshot"]
        script = (f"import sys; import {', '.join(modules)}; "
                  "print(','.join(sorted(m for m in sys.modules if m.startswith(('ax_g_ai.services', 'ax_g_ai.adapters')))))")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "")

    def test_api_does_not_import_the_composition_root(self) -> None:
        self.assert_no_imports(module_files("api"), ("ax_g_ai.runtime", "ax_g_ai.config"))


if __name__ == "__main__":
    unittest.main()
