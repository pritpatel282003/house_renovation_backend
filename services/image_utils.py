import io

from PIL import Image

from services.http_client import get_http_client


async def download_image(url: str, mode: str = "RGBA", timeout: float = 30.0) -> Image.Image:
    client = get_http_client()
    resp = await client.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert(mode)
