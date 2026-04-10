from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import os
import uuid
from nltk.tokenize import sent_tokenize
import nltk
import spacy
from spacy.matcher import PhraseMatcher
from parser import parse_resume, build_summary

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load spaCy
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

SKILLS_BY_CATEGORY = {
    "Programming":  ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift"],
    "Web":          ["fastapi", "flask", "django", "react", "nodejs", "express", "html", "css"],
    "Data & ML":    ["machine learning", "deep learning", "tensorflow", "pytorch", "keras", "pandas", "numpy", "scikit-learn"],
    "Cloud":        ["aws", "gcp", "azure", "docker", "kubernetes", "terraform"],
    "Databases":    ["sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch"],
    "Tools":        ["git", "linux", "jira", "jenkins", "airflow", "spark"],
}

for category, skills in SKILLS_BY_CATEGORY.items():
    patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add(category, patterns)


# 📄 Extract text from PDF
# Strategy:
#   1. Try pdfplumber (best for structured/text PDFs)
#   2. Fallback to pymupdf/fitz (handles more PDF variants)
#   3. If both yield nothing → scanned/image PDF
#   Returns: (text, error_type)
#     error_type = None | "corrupted" | "scanned"
def extract_pdf_text(file_path):
    text = ""

    # ── Attempt 1: pdfplumber ──────────────────────────
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:
                    continue
    except Exception:
        text = ""  # pdfplumber failed, try fitz below

    # ── Attempt 2: pymupdf (fitz) fallback ────────────
    if not text.strip():
        try:
            import fitz  # pymupdf
            doc = fitz.open(file_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
        except fitz.FileDataError:
            return None, "corrupted"   # genuinely broken file
        except Exception:
            return None, "corrupted"

    if not text.strip():
        return None, "scanned"         # both extractors found no text

    return text.strip(), None


# 🧹 Clean text (lowercase, collapse whitespace)
def clean_text(text):
    return " ".join(text.lower().split())


# 🧠 Summary — 50 words max, from ORIGINAL cased text
def summarize(text, max_words=50):
    sentences = sent_tokenize(text)
    summary_words = []
    for sentence in sentences:
        words = sentence.split()
        if len(summary_words) + len(words) <= max_words:
            summary_words.extend(words)
        else:
            remaining = max_words - len(summary_words)
            if remaining > 0:
                summary_words.extend(words[:remaining])
            break
    result = " ".join(summary_words).strip()
    return (result + "...") if result else ""


# 🎯 Extract skills using spaCy PhraseMatcher
def extract_skills(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    found = {}
    for match_id, start, end in matches:
        category = nlp.vocab.strings[match_id]
        skill = doc[start:end].text
        if category not in found:
            found[category] = []
        if skill not in found[category]:
            found[category].append(skill)
    return found


# 🔍 Detect document type
def detect_type(text):
    if "revenue" in text or "profit" in text:
        return "financial"
    return "resume"


# 🔥 API Endpoint
@app.post("/parse")
async def parse(file: UploadFile = File(...)):

    # ✅ Validate file extension
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    # ✅ Use UUID to avoid race conditions with concurrent uploads
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        with open(file_path, "wb") as f:
            f.write(contents)

        # ✅ Extract text (two-stage: pdfplumber → pymupdf fallback)
        text, error_type = extract_pdf_text(file_path)

        if error_type in ("corrupted", "scanned"):
            raise HTTPException(
                status_code=422,
                detail="Only text-based (readable) PDFs are supported. "
                       "Please upload a PDF with selectable text, not a scanned image or image-only file."
            )

        # ✅ Skills from lowercased text
        clean = clean_text(text)
        skills = extract_skills(clean)
        doc_type = detect_type(clean)

        # ✅ Structured resume parsing
        parsed_data = parse_resume(text)

        # ✅ Smart summary: uses resume's own summary section, or auto-generates
        summary = build_summary(
            name             = parsed_data.get("name"),
            years_exp        = parsed_data.get("years_of_experience", 0),
            companies        = parsed_data.get("companies", []),
            skills_by_category = skills,
            sections         = parsed_data.get("_raw_sections", {}),
        )
        # Remove internal key before sending response
        parsed_data.pop("_raw_sections", None)

    finally:
        # ✅ Always clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "status": "success",
        "data": parsed_data,
        "document_type": doc_type,
        "summary": summary,
        "skills_by_category": skills
    }