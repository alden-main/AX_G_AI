def main() -> None:
    import uvicorn

    uvicorn.run("ax_g_ai.api:app", host="0.0.0.0", port=8100)
