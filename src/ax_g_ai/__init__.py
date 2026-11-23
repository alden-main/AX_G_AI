"""AX-G AI 임상 채팅 서비스 패키지와 ``ax-g-ai`` 실행 명령의 진입점.

설계상 위치: 패키지 최상위. ``pyproject.toml``의 ``ax-g-ai = "ax_g_ai:main"`` 스크립트가 ``main``을 호출한다.

역할: Uvicorn으로 ``ax_g_ai.runtime:create_runtime_app`` 팩토리 앱을 ``0.0.0.0:8100``에서 띄운다.
"""


def main() -> None:
    """``create_runtime_app`` 팩토리로 만든 API를 Uvicorn으로 실행한다(포트 8100)."""
    import uvicorn

    uvicorn.run("ax_g_ai.runtime:create_runtime_app", factory=True, host="0.0.0.0", port=8100)
