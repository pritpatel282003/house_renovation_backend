import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)


class Settings:
    SERVICE_SECRET: str = os.getenv("SERVICE_SECRET", "")
    ROBOFLOW_API_KEY: str = os.getenv("ROBOFLOW_API_KEY", "")
    ROBOFLOW_MODEL_ID: str = os.getenv("ROBOFLOW_MODEL_ID", "")
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_IMAGE_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT", "")

    @property
    def roboflow_configured(self) -> bool:
        return bool(self.ROBOFLOW_API_KEY and self.ROBOFLOW_MODEL_ID)


settings = Settings()
