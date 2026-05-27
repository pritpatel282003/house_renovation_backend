import logging

from fastapi import APIRouter, HTTPException

from schemas.segment import SegmentRequest, SegmentResponse
from services.sam_service import segment_image

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    try:
        segments = await segment_image(req.image_url)
        return SegmentResponse(segments=segments)
    except Exception as exc:
        logger.exception("Segmentation failed for project %s", req.project_id)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")
