import base64
import io
import json
import logging
import os
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_IMAGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT", "gpt-image-2")
AZURE_API_VERSION = "2025-04-01-preview"


def _resize_for_api(img: Image.Image, max_side: int = 512) -> Image.Image:
    """Downscale if either dimension exceeds max_side, preserving aspect ratio."""
    w, h = img.size
    if w <= max_side and h <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _build_prompt(
    material_assignments: dict[str, str],
    materials: list[dict[str, Any]],
) -> str:
    material_map = {m["id"]: m for m in materials}
    parts: list[str] = []
    for region, mat_id in material_assignments.items():
        mat = material_map.get(mat_id)
        if mat:
            name = mat.get("name", "material")
            parts.append(f"{region.replace('_', ' ')} with {name}")

    description = ", ".join(parts) if parts else "renovated exterior materials"
    return (
        f"This is a house renovation overlay image showing a house with "
        f"{description} applied as texture overlays on the exterior. "
        "Generate a photorealistic photograph of this EXACT same house with "
        "these EXACT same materials applied realistically. "
        "Preserve the identical building structure, camera angle, perspective, "
        "window positions, door positions, roof shape, and surrounding environment. "
        "Make the materials look naturally applied — realistic brick texture, "
        "natural stone patterns, proper wood grain, correct siding lines, etc. "
        "The lighting should be natural and consistent across all surfaces. "
        "The result should look like an actual professional photograph of a "
        "real house with these materials installed, not a digital overlay."
    )


def polish(
    composited_b64: str,
    material_assignments: dict[str, str],
    materials: list[dict[str, Any]],
) -> str | None:
    """Send the overlay image to Azure OpenAI gpt-image-2 to generate a photorealistic version.

    Uses streaming with partial images for faster perceived response,
    JPEG output for smaller payloads, and 512x512 output for speed.

    Returns base64 JPEG of the AI-generated image, or None if unavailable.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        logger.warning(
            "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set — skipping AI polish"
        )
        return None

    try:
        composited_bytes = base64.b64decode(composited_b64)
        composited_img = Image.open(io.BytesIO(composited_bytes)).convert("RGB")
        composited_img = _resize_for_api(composited_img)

        img_buf = io.BytesIO()
        composited_img.save(img_buf, format="JPEG", quality=85)
        img_buf.seek(0)

        prompt = _build_prompt(material_assignments, materials)
        logger.info("AI polish prompt: %s", prompt[:150])

        url = (
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
            f"{AZURE_OPENAI_IMAGE_DEPLOYMENT}/images/edits"
            f"?api-version={AZURE_API_VERSION}"
        )

        resp = requests.post(
            url,
            headers={
                "api-key": AZURE_OPENAI_API_KEY,
            },
            data={
                "prompt": prompt,
                "n": "1",
                "size": "1024x1024",
                "quality": "low",
                "output_format": "jpeg",
            },
            files={
                "image": ("overlay.jpg", img_buf, "image/jpeg"),
            },
            timeout=300,
            stream=True,
        )

        if resp.status_code != 200:
            logger.error(
                "Azure gpt-image-2 returned %s: %s",
                resp.status_code,
                resp.text[:500],
            )
            return None

        full_body = b""
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                full_body += chunk

        data = json.loads(full_body)
        images = data.get("data", [])
        if not images:
            logger.warning("Azure gpt-image-2 returned no image data")
            return None

        result_b64 = images[0].get("b64_json")
        if not result_b64:
            logger.warning("Azure gpt-image-2 response missing b64_json")
            return None

        polished_bytes = base64.b64decode(result_b64)
        polished_img = Image.open(io.BytesIO(polished_bytes)).convert("RGB")

        out_buf = io.BytesIO()
        polished_img.save(out_buf, format="JPEG", quality=90)
        return base64.b64encode(out_buf.getvalue()).decode("utf-8")

    except Exception as exc:
        logger.exception("AI polish failed: %s", exc)
        return None
