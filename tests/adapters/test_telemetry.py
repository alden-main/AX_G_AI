from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ax_g_ai.adapters.telemetry import instrumentation_from_environment


class InstrumentationFromEnvironmentTest(unittest.TestCase):
    def test_tracing_stays_off_without_an_otlp_endpoint(self) -> None:
        for environment in ({}, {"OTEL_EXPORTER_OTLP_ENDPOINT": " "}, {"OTEL_SERVICE_NAME": "ax-g-ai"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertIsNone(instrumentation_from_environment(include_content=True))


if __name__ == "__main__":
    unittest.main()
