from pydantic import BaseModel


class ValidateRequest(BaseModel):
    image_url: str
