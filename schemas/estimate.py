from typing import Any

from pydantic import BaseModel, Field


class EstimateRequest(BaseModel):
    image_url: str
    segmentation_data: list[dict[str, Any]]
    material_assignments: dict[str, str]
    materials: list[dict[str, Any]]
    rate_overrides: dict[str, dict[str, float]] | None = None
    wastage_percent: float = Field(default=12.0)
    pixels_per_foot: float | None = None


class EstimateResponse(BaseModel):
    area_data: dict[str, Any]
    quantity_data: dict[str, Any]
    cost_data: dict[str, Any]
