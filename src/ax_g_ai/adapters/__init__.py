"""외부 시스템과 내부 도메인 사이의 경계."""

from .emr import EmrAdapter, EmrAdapterError, EmrReadClient, EmrReadRequest, UrllibEmrReadClient
from .ai_tank import AiTankClient, AiTankProviderError, ChatHistoryItem, ConsultRequest
from .main_bridge import MainBridgePayloadMapper

__all__ = [
    "AiTankClient",
    "AiTankProviderError",
    "ChatHistoryItem",
    "ConsultRequest",
    "MainBridgePayloadMapper",
    "EmrAdapter",
    "EmrAdapterError",
    "EmrReadClient",
    "EmrReadRequest",
    "UrllibEmrReadClient",
]
