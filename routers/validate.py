from fastapi import APIRouter
from pydantic import BaseModel
from services.validation_service import check_blur

router = APIRouter()


class ValidateRequest(BaseModel):
    image_url: str


@router.post("/blur")
async def validate_blur(body: ValidateRequest):
    result = check_blur(body.image_url)
    return result
