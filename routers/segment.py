import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.sam_service import segment_image

logger = logging.getLogger(__name__)
router = APIRouter()


class SegmentRequest(BaseModel):
    image_url: str
    project_id: str


class SegmentResponse(BaseModel):
    segments: list[dict[str, Any]]


@router.post("", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    try:
        segments = segment_image(req.image_url)
        return SegmentResponse(segments=segments)
    except Exception as exc:
        logger.exception("Segmentation failed for project %s", req.project_id)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")
