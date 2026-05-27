from typing import Any

from pydantic import BaseModel


class VisualizeRequest(BaseModel):
    image_url: str
    segmentation_data: list[dict[str, Any]]
    material_assignments: dict[str, str]
    materials: list[dict[str, Any]]


class VisualizeResponse(BaseModel):
    redesigned_image_base64: str
