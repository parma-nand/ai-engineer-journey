from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import os
from nltk.tokenize import sent_tokenize
import nltk
import spacy
from spacy.matcher import PhraseMatcher
from parser import parse_resume
from pdfminer.high_level import extract_text

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load spaCy FIRST
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

ALL_SKILLS = []
for category, skills in SKILLS_BY_CATEGORY.items():
    patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add(category, patterns)
    ALL_SKILLS.extend(skills)

# 📄 Extract text from PDF
def extract_text(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception:
                    continue
    except Exception:
        return None  # ✅ fake or corrupted PDF returns None
    return text.strip()

# 🧹 Clean text
def clean_text(text):
    return " ".join(text.lower().split())

# 🧠 Summary
def summarize(text, max_chars=300):
    sentences = sent_tokenize(text)
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) <= max_chars:
            summary += sentence + " "
        else:
            break
    return summary.strip() if summary else sentences[0][:max_chars]

# 🎯 Extract skills
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

# 🔥 API

@app.post("/parse")
async def parse(file: UploadFile = File(...)):

    # ✅ Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # ✅ Unique temp file (IMPORTANT)
    file_path = os.path.join(UPLOAD_FOLDER, f"temp_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(contents)

    # ✅ Extract text
    text = extract_text(file_path)

    if text is None:
        raise HTTPException(status_code=422, detail="Corrupted PDF")

    if text.strip() == "":
        raise HTTPException(status_code=422, detail="Scanned/image PDF not supported")

    # ✅ Clean + reuse logic
    clean = clean_text(text)

    # ✅ OLD features
    summary = summarize(clean)
    skills = extract_skills(clean)
    doc_type = detect_type(clean)

    # ✅ NEW parsing
    parsed_data = parse_resume(text)

    return {
        "status": "success",
        "data": parsed_data,
        "document_type": doc_type,
        "summary": summary,
        "skills_by_category": skills
    }