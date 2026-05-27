import logging

from fastapi import APIRouter, HTTPException

from schemas.ai_design import AiDesignRequest, AiDesignResponse
from services.visualization_service import visualize as run_visualization
from services.ai_polish_service import polish as run_polish

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AiDesignResponse)
async def ai_design(req: AiDesignRequest):
    try:
        overlay_b64 = await run_visualization(
            image_url=req.image_url,
            segmentation_data=req.segmentation_data,
            material_assignments=req.material_assignments,
            materials=req.materials,
        )

        polished_b64 = await run_polish(
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
