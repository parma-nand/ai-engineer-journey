import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.parser_service import ResumeParserService
from app.models.resume import ParsedResume
from app.config import get_settings

settings = get_settings()
router = APIRouter()
parser_service = ResumeParserService()

# In-memory result cache (replace with Redis/DB for production)
_result_cache: dict[str, ParsedResume] = {}


@router.post("/upload", response_model=ParsedResume, summary="Upload and parse a resume PDF")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF resume and receive structured parsed data.
    - Max size: 5MB
    - Max pages: 5
    - PDF must be text-based (not scanned)
    """

    # ── Validation ────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Invalid content type. Upload a PDF.")

    contents = await file.read()

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB."
        )

    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted.")

    # ── Save temp file ────────────────────────────────────────────
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    tmp_name = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(settings.UPLOAD_DIR, tmp_name)

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(contents)

        logger.info(f"Processing resume: {file.filename} ({len(contents)//1024}KB)")

        # ── Parse ─────────────────────────────────────────────────
        result = parser_service.parse(tmp_path)

        # Cache result
        _result_cache[result.id] = result

        logger.info(f"Parsed successfully. ID={result.id}, score={result.score.overall}")
        return result

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error parsing resume: {e}")
        raise HTTPException(status_code=500, detail="Internal error while parsing resume.")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/result/{result_id}", response_model=ParsedResume, summary="Fetch a previously parsed resume")
async def get_result(result_id: str):
    """Retrieve a previously parsed resume result by ID."""
    result = _result_cache.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found. Please re-upload the resume.")
    return result


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}