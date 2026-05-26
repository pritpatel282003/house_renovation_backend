import asyncio
import io
import logging
from typing import Any

import cv2
import numpy as np

from services.http_client import get_http_client

logger = logging.getLogger(__name__)

BLUR_THRESHOLD = 100.0


def _analyze_blur(content: bytes) -> dict:
    img_array = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"is_blurry": False, "score": 0, "error": "Could not decode image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "is_blurry": bool(laplacian_var < BLUR_THRESHOLD),
        "score": float(round(laplacian_var, 2)),
        "threshold": float(BLUR_THRESHOLD),
    }


async def check_blur(image_url: str) -> dict:
    """Download image and check if it's blurry using Laplacian variance."""
    try:
        client = get_http_client()
        resp = await client.get(image_url, timeout=15.0)
        resp.raise_for_status()
        return await asyncio.to_thread(_analyze_blur, resp.content)
    except Exception as exc:
        return {"is_blurry": False, "score": 0, "error": str(exc)}
