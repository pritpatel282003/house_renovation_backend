import logging

from fastapi import APIRouter, HTTPException

from schemas.visualize import VisualizeRequest, VisualizeResponse
from services.visualization_service import visualize as run_visualization

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=VisualizeResponse)
async def visualize(req: VisualizeRequest):
    try:
        result_b64 = await run_visualization(
            image_url=req.image_url,
            segmentation_data=req.segmentation_data,
            material_assignments=req.material_assignments,
            materials=req.materials,
        )
        return VisualizeResponse(redesigned_image_base64=result_b64)
    except Exception as exc:
        logger.exception("Visualization failed")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {exc}")
