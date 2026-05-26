import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.visualization_service import visualize as run_visualization
from services.ai_polish_service import polish as run_polish

logger = logging.getLogger(__name__)
router = APIRouter()


class AiDesignRequest(BaseModel):
    image_url: str
    segmentation_data: list[dict[str, Any]]
    material_assignments: dict[str, str]
    materials: list[dict[str, Any]]


class AiDesignResponse(BaseModel):
    overlay_image_base64: str
    ai_polished_image_base64: str | None


@router.post("", response_model=AiDesignResponse)
async def ai_design(req: AiDesignRequest):
    try:
        overlay_b64 = run_visualization(
            image_url=req.image_url,
            segmentation_data=req.segmentation_data,
            material_assignments=req.material_assignments,
            materials=req.materials,
        )

        polished_b64 = run_polish(
            composited_b64=overlay_b64,
            material_assignments=req.material_assignments,
            materials=req.materials,
        )

        return AiDesignResponse(
            overlay_image_base64=overlay_b64,
            ai_polished_image_base64=polished_b64,
        )
    except Exception as exc:
        logger.exception("AI design failed")
        raise HTTPException(status_code=500, detail=f"AI design failed: {exc}")
