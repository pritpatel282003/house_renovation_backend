from typing import Any

from pydantic import BaseModel


class SegmentRequest(BaseModel):
    image_url: str
    project_id: str


class SegmentResponse(BaseModel):
    segments: list[dict[str, Any]]
