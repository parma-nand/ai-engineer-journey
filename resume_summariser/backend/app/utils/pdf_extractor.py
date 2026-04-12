import pdfplumber
from loguru import logger
from typing import Tuple, Optional


def extract_text_from_pdf(file_path: str) -> Tuple[Optional[str], Optional[str], int]:
    """
    Extract text from PDF using pdfplumber → PyMuPDF fallback.
    Returns: (text, error_type, page_count)
      error_type: None | "corrupted" | "scanned" | "empty"
    """
    text = ""
    page_count = 0

    # ── Attempt 1: pdfplumber ─────────────────────────────────────
    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                try:
                    # Handle multi-column layouts by using bounding boxes
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"pdfplumber page error: {e}")
                    continue
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        text = ""

    # ── Attempt 2: PyMuPDF fallback ───────────────────────────────
    if not text.strip():
        try:
            import fitz
            doc = fitz.open(file_path)
            page_count = doc.page_count
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}")
            return None, "corrupted", 0

    if not text.strip():
        return None, "scanned", page_count

    # Basic sanity: if text is very short it might be mostly images
    if len(text.strip()) < 100:
        return None, "scanned", page_count

    return text.strip(), None, page_count