"""환경 설정으로 Bridge API 런타임을 조립한다."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.api import create_app
from ax_g_ai.config import AISettings, ConfigurationError, RuntimeSettings
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.patient_context_store import PatientContextStore
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup

logger = logging.getLogger("uvicorn.error")


def create_runtime_app() -> FastAPI:
    """필수 설정이 없거나 잘못되면 AI를 호출하지 않는 fail-closed 앱을 생성한다."""
    try:
        ai_settings = AISettings.from_environment()
        runtime_settings = RuntimeSettings.from_environment()
    except ConfigurationError as error:
        # The message names fields and reasons only, never configured values.
        logger.warning("runtime_configuration_invalid reason=%s", error)
        return create_app()
    database_path = runtime_settings.reference_data_db_path
    lookup = None if database_path is None else SqliteReferenceLookup(database_path)
    return create_app(
        ClinicalChatService(build_provider_client(ai_settings), lookup),
        RawEmrPayloadMapper(lookup),
        PatientContextStore(
            runtime_settings.patient_context_ttl_seconds,
            runtime_settings.patient_context_max_entries,
        ),
    )
