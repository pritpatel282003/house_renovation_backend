from typing import Any

from pydantic import BaseModel


class AiDesignRequest(BaseModel):
    image_url: str
    segmentation_data: list[dict[str, Any]]
    material_assignments: dict[str, str]
    materials: list[dict[str, Any]]


class AiDesignResponse(BaseModel):
    overlay_image_base64: str
    ai_polished_image_base64: str | None
