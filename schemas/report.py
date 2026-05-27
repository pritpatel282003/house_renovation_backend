from typing import Any

from pydantic import BaseModel


class ReportRequest(BaseModel):
    model_config = {"extra": "allow"}

    title: str = "Untitled Project"
    original_image_url: str | None = None
    redesigned_image_url: str | None = None
    cost_data: dict[str, Any] = {}
