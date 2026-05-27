from fastapi import APIRouter

from schemas.validate import ValidateRequest
from services.validation_service import check_blur

router = APIRouter()


@router.post("/blur")
async def validate_blur(body: ValidateRequest):
    result = await check_blur(body.image_url)
    return result
