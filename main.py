import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import settings
from core.middleware import register_middleware
from routers import segment, visualize, estimate, report, validate, ai_design
from services.http_client import close_http_client, init_http_client

logger = logging.getLogger("renovation-service")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    if settings.roboflow_configured:
        logger.info("Roboflow API configured — model: %s", settings.ROBOFLOW_MODEL_ID)
    else:
        logger.warning(
            "ROBOFLOW_API_KEY or ROBOFLOW_MODEL_ID not set — "
            "segmentation will use mock fallback"
        )
    yield
    await close_http_client()


app = FastAPI(title="Renovation AI Service", version="1.0.0", lifespan=lifespan)

register_middleware(app)


@app.get("/health")
async def health():
    return {
        "roboflow_configured": settings.roboflow_configured,
        "model_id": settings.ROBOFLOW_MODEL_ID or None,
        "version": "1.0.0",
    }


app.include_router(validate.router, prefix="/validate", tags=["Validation"])
app.include_router(segment.router, prefix="/segment", tags=["Segmentation"])
app.include_router(visualize.router, prefix="/visualize", tags=["Visualization"])
app.include_router(estimate.router, prefix="/estimate", tags=["Estimation"])
app.include_router(report.router, prefix="/report", tags=["Report"])
app.include_router(ai_design.router, prefix="/ai-design", tags=["AI Design"])
