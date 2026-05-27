import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from schemas.estimate import EstimateRequest, EstimateResponse
from services.area_service import calculate_areas
from services.cost_service import calculate_costs

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=EstimateResponse)
async def estimate(req: EstimateRequest):
    try:
        area_kwargs: dict[str, Any] = {
            "image_url": req.image_url,
            "segmentation_data": req.segmentation_data,
        }
        if req.pixels_per_foot is not None:
            area_kwargs["pixels_per_foot"] = req.pixels_per_foot

        area_data = await calculate_areas(**area_kwargs)

        cost_data = await calculate_costs(
            area_data=area_data,
            material_assignments=req.material_assignments,
            materials=req.materials,
            rate_overrides=req.rate_overrides,
            wastage_percent=req.wastage_percent,
        )

        quantity_data: dict[str, Any] = {}
        for item in cost_data.get("line_items", []):
            quantity_data[item["region"]] = {
                "quantity": item["quantity"],
                "unit": item["unit"],
            }

        return EstimateResponse(
            area_data=area_data,
            quantity_data=quantity_data,
            cost_data=cost_data,
        )
    except Exception as exc:
        logger.exception("Estimation failed")
        raise HTTPException(status_code=500, detail=f"Estimation failed: {exc}")
