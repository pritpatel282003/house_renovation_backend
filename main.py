import os
import logging
from contextlib import asynccontextmanager

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import segment, visualize, estimate, report, validate, ai_design

from services.http_client import close_http_client, init_http_client

logger = logging.getLogger("renovation-service")
logging.basicConfig(level=logging.INFO)

SERVICE_SECRET = os.getenv("SERVICE_SECRET", "")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    if ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID:
        logger.info("Roboflow API configured — model: %s", ROBOFLOW_MODEL_ID)
    else:
        logger.warning(
            "ROBOFLOW_API_KEY or ROBOFLOW_MODEL_ID not set — "
            "segmentation will use mock fallback"
        )
    yield
    await close_http_client()


app = FastAPI(title="Renovation AI Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}

@app.middleware("http")
async def verify_service_secret(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    token = request.headers.get("x-service-secret", "")
    if SERVICE_SECRET and token != SERVICE_SECRET:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


@app.get("/health")
async def health():
    return {
        "roboflow_configured": bool(ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID),
        "model_id": ROBOFLOW_MODEL_ID or None,
        "version": "1.0.0",
    }


app.include_router(validate.router, prefix="/validate", tags=["Validation"])
app.include_router(segment.router, prefix="/segment", tags=["Segmentation"])
app.include_router(visualize.router, prefix="/visualize", tags=["Visualization"])
app.include_router(estimate.router, prefix="/estimate", tags=["Estimation"])
app.include_router(report.router, prefix="/report", tags=["Report"])
app.include_router(ai_design.router, prefix="/ai-design", tags=["AI Design"])
