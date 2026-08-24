import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import analyze, health, streaming
from app.core.error_handlers import (
    analysis_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from app.core.exceptions import AnalysisError
from app.ml.model_manager import load_model
from app.utils.middleware import RequestIDMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("voice_analyzer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Voice Gender/Age/Quality Analyzer Service...")
    try:
        load_model()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load model during startup: {e}")
    yield
    logger.info("Shutting down Voice Gender/Age/Quality Analyzer Service...")


app = FastAPI(
    title="Voice Gender/Age/Quality Analyzer",
    description="Microservice for audio quality classification & speaker gender/age estimation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow_credentials must be False with wildcard origin (fixes WebSocket 403)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware
app.add_middleware(RequestIDMiddleware)

# Exception Handlers
app.add_exception_handler(AnalysisError, analysis_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Routers
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(streaming.router)
