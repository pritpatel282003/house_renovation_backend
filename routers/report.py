import logging
from io import BytesIO
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.report_service import generate_report

logger = logging.getLogger(__name__)
router = APIRouter()


class ReportRequest(BaseModel):
    model_config = {"extra": "allow"}

    title: str = "Untitled Project"
    original_image_url: str | None = None
    redesigned_image_url: str | None = None
    cost_data: dict[str, Any] = {}


@router.post("")
async def report(req: ReportRequest):
    try:
        project_data = req.model_dump()
        pdf_bytes = await generate_report(project_data)
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=renovation_report.pdf"},
        )
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")
