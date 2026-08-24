import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AnalysisError
from app.schemas.analysis import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
    logger.warning(
        f"AnalysisError [{exc.code}]: {exc.message} (path: {request.url.path})"
    )
    response_data = ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message))
    return JSONResponse(status_code=exc.status_code, content=response_data.model_dump())


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    msg = (
        errors[0].get("msg", "Invalid request parameters.")
        if errors
        else "Invalid request parameters."
    )
    logger.warning(f"RequestValidationError: {msg} (path: {request.url.path})")
    response_data = ErrorResponse(
        error=ErrorDetail(code="INVALID_REQUEST", message=msg)
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response_data.model_dump(),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    logger.warning(
        f"HTTPException [{exc.status_code}]: {exc.detail} (path: {request.url.path})"
    )
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    response_data = ErrorResponse(error=ErrorDetail(code=code, message=str(exc.detail)))
    return JSONResponse(status_code=exc.status_code, content=response_data.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception: {exc}")
    response_data = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_ERROR",
            message="An unexpected internal server error occurred.",
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data.model_dump(),
    )
