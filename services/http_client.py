import httpx

_client: httpx.AsyncClient | None = None


async def init_http_client() -> None:
    global _client
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
    )


async def close_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_http_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("HTTP client not initialized — call init_http_client() first")
    return _client
