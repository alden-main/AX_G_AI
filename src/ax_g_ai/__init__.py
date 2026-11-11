def main() -> None:
    import uvicorn

    uvicorn.run("ax_g_ai.runtime:create_runtime_app", factory=True, host="0.0.0.0", port=8100)
