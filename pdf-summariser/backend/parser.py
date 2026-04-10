import re
import spacy
from dateutil import parser as date_parser
from datetime import datetime

nlp = spacy.load("en_core_web_sm")


# ──────────────────────────────────────────
# 📧 Email
# ──────────────────────────────────────────
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


# ──────────────────────────────────────────
# 📞 Phone
# ──────────────────────────────────────────
def extract_phone(text):
    match = re.search(r"[\(\+]?\d[\d\s\-\(\)\.]{7,15}\d", text)
    if not match:
        return None
    phone = match.group(0).strip()
    phone = re.sub(r"[\(\)]", "", phone)
    phone = re.sub(r"\s+", " ", phone).strip()
    if re.match(r"^\+\d+$", phone):
        digits = phone[1:]
        for cc_len in [1, 2, 3]:
            local = digits[cc_len:]
            if len(local) == 10:
                phone = "+" + digits[:cc_len] + " " + local
                break
    return phone


# ──────────────────────────────────────────
# 👤 Name
# ──────────────────────────────────────────
_HEADING_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary",
    "objective", "contact", "address", "email", "phone",
    "linkedin", "github", "education", "experience", "skills",
    "projects", "achievements", "certifications", "declaration",
    "professional", "technical", "languages", "interests",
}

_CONTACT_PATTERNS = re.compile(
    r"@|linkedin|github|http|www\.|"
    r"\+?\(?\d{3,}|"
    r"\|",
    re.IGNORECASE
)

def _is_contact_line(line):
    return bool(_CONTACT_PATTERNS.search(line))

def extract_name(text):
    all_lines = [l.strip() for l in text.split("\n") if l.strip()][:6]
    candidate_lines = [l for l in all_lines if not _is_contact_line(l)]

    for line in candidate_lines:
        titled = line.title()
        doc = nlp(titled)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if not re.search(r"\d", name) and 2 <= len(name.split()) <= 5:
                    return titled if line.isupper() else name

    for line in candidate_lines:
        words = line.split()
        if not (2 <= len(words) <= 4):
            continue
        if not all(re.fullmatch(r"[A-Za-z]+\.?", w) for w in words):
            continue
        if not all(w[0].isupper() for w in words):
            continue
        lower_words = {w.lower() for w in words}
        if lower_words.intersection(_HEADING_WORDS):
            continue
        return line.title() if line.isupper() else line

    return None


# ──────────────────────────────────────────
# ✂️ Limit text to N words
# ──────────────────────────────────────────
def limit_words(text, max_words=50):
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


# ──────────────────────────────────────────
# 📂 Section Splitter
#    Added "summary" as a tracked section
# ──────────────────────────────────────────
SECTION_KEYWORDS = {
    "summary":        ["summary", "objective", "profile", "about me",
                       "professional summary", "executive summary", "career objective"],
    "experience":     ["experience", "work history", "employment", "professional experience"],
    "education":      ["education", "academic", "qualification"],
    "projects":       ["project", "projects", "personal projects", "key projects"],
    "certifications": ["certification", "certifications", "certificate", "certificates",
                       "courses", "training", "seminar"],
}

def detect_section(line_lower):
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', line_lower):
                return section
    return None

def extract_sections(text):
    sections = {k: "" for k in SECTION_KEYWORDS}
    current = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                sections[current] += "\n"
            continue
        detected = detect_section(stripped.lower())
        if detected:
            current = detected
            continue
        if current:
            sections[current] += stripped + "\n"

    return {k: v.strip() for k, v in sections.items()}


# ──────────────────────────────────────────
# 🏢 Company Extractor
#    Primary:  spaCy ORG NER
#    Fallback: lines with company suffix keywords
# ──────────────────────────────────────────
_COMPANY_SUFFIXES = re.compile(
    r"\b(pvt\.?\s*ltd\.?|ltd\.?|llc|inc\.?|corp\.?|technologies|tech|solutions|"
    r"services|systems|consulting|software|global|group|foundation|ventures|"
    r"associates|enterprises|limited)\b",
    re.IGNORECASE
)

_DATE_RANGE_PAT = re.compile(
    r"(\d{1,2}/\d{4}|[A-Za-z]{3,9}\.?\s?\d{4}|\d{4}[-/][A-Za-z]{3,9}|[A-Za-z]{3,9}[-/]\d{4}|\d{4})"
    r"\s*[-–—to]+\s*"
    r"(\d{1,2}/\d{4}|[A-Za-z]{3,9}\.?\s?\d{4}|\d{4}[-/][A-Za-z]{3,9}|[A-Za-z]{3,9}[-/]\d{4}|\d{4}|present|current|now)",
    re.IGNORECASE
)

# Known tech terms spaCy wrongly tags as ORG
_TECH_TERMS = {
    "java", "spring", "spring boot", "react", "react.js", "node", "nodejs",
    "html", "css", "sql", "aws", "gcp", "azure", "docker", "kubernetes",
    "python", "django", "flask", "fastapi", "hibernate", "jpa", "jdbc",
    "bootstrap", "javascript", "typescript", "postman", "git", "github",
    "mysql", "postgresql", "mongodb", "redis", "kafka", "jenkins", "maven",
    "swagger", "mockito", "selenium", "agile", "rest", "restful", "api",
    "linux", "unix", "jira", "terraform", "microservices",
}

def _get_header_lines(exp_text):
    """Only job-header lines — skip bullets and long description sentences."""
    result = []
    for line in exp_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("\u2022", "-", "*", "\u2013", "+")):
            continue
        if len(stripped.split()) > 8 and not _DATE_RANGE_PAT.search(stripped):
            continue
        result.append(stripped)
    return "\n".join(result)

def _is_valid_company(name):
    """Reject NER ORG results that are clearly not company names."""
    if not name or len(name.split()) > 6:
        return False
    if any(c in name for c in ("\u2022", "\u2013", "\u2014", "%")):
        return False
    if name[0].islower():
        return False
    if name.lower().strip() in _TECH_TERMS:
        return False
    return True

def extract_companies(exp_text):
    if not exp_text:
        return []

    companies = []

    # Primary: spaCy ORG NER on header lines only (not bullet descriptions)
    header_text = _get_header_lines(exp_text)
    doc = nlp(header_text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            name = ent.text.strip()
            if _is_valid_company(name) and name not in companies:
                companies.append(name)

    if companies:
        return companies

    # Fallback: lines containing company suffix keywords
    for line in exp_text.split("\n"):
        line = line.strip()
        if not line or line.startswith(("\u2022", "-", "*", "+")):
            continue
        if _COMPANY_SUFFIXES.search(line):
            clean = _DATE_RANGE_PAT.sub("", line)
            clean = re.sub(r"\|.*", "", clean)
            clean = re.sub(r"\(.*?\)", "", clean)
            clean = re.sub(r",\s*[A-Z][a-z].*$", "", clean)
            clean = clean.strip(" ,-")
            if clean and len(clean) > 2 and clean not in companies:
                companies.append(clean)

    return companies



# ──────────────────────────────────────────
# 📅 Years of Experience
# ──────────────────────────────────────────
def extract_years_of_experience(exp_text):
    if not exp_text:
        return 0

    _DT = (
        r"(?:\d{1,2}/\d{4}"
        r"|[A-Za-z]{3,9}\.?\s?\d{4}"
        r"|\d{4}[-/][A-Za-z]{3,9}"
        r"|[A-Za-z]{3,9}[-/]\d{4}"
        r"|\d{4})"
    )
    _END = (
        r"(?:\d{1,2}/\d{4}"
        r"|[A-Za-z]{3,9}\.?\s?\d{4}"
        r"|\d{4}[-/][A-Za-z]{3,9}"
        r"|[A-Za-z]{3,9}[-/]\d{4}"
        r"|\d{4}"
        r"|present|current|now)"
    )
    range_pattern = re.compile(
        r"(" + _DT + r")\s*[-–—to]+\s*(" + _END + r")",
        re.IGNORECASE
    )

    intervals = []
    for m in range_pattern.finditer(exp_text):
        start_str, end_str = m.group(1), m.group(2)
        try:
            start_norm = re.sub(r"[-/]", " ", start_str)
            end_norm   = re.sub(r"[-/]", " ", end_str)
            start = date_parser.parse(start_norm, default=date_parser.parse("Jan 2000"))
            if re.match(r"present|current|now", end_str, re.IGNORECASE):
                end = datetime.today()
            else:
                end = date_parser.parse(end_norm, default=date_parser.parse("Jan 2000"))
            if end > start:
                intervals.append((start, end))
        except Exception:
            continue

    if not intervals:
        years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", exp_text)]
        if len(years) >= 2:
            return round(max(years) - min(years), 1)
        return 0

    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 365, 1)


# ──────────────────────────────────────────
# 📝 Smart Summary
#    Priority 1: Resume's own Summary/Objective section (<=50 words)
#    Priority 2: Auto-generate from parsed fields
# ──────────────────────────────────────────
def build_summary(name, years_exp, companies, skills_by_category, sections):

    # Priority 1: use the candidate's own summary section
    own_summary = sections.get("summary", "").strip()
    if own_summary:
        return limit_words(own_summary, max_words=50)

    # Priority 2: generate from parsed data
    parts = []

    # Job title from first line of experience
    exp_text = sections.get("experience", "")
    job_title = None
    if exp_text:
        first_line = exp_text.split("\n")[0].strip()
        title_candidate = _DATE_RANGE_PAT.sub("", first_line).strip(" ,–-|")
        if title_candidate and not title_candidate.startswith(("*", "+")) and len(title_candidate.split()) <= 6:
            job_title = title_candidate

    if name and job_title:
        parts.append(f"{name} is a {job_title}")
    elif name:
        parts.append(f"{name} is a professional")

    if years_exp and years_exp > 0:
        parts.append(f"with {years_exp} years of experience.")
    elif parts:
        parts[-1] += "."

    if companies:
        company_str = " and ".join(companies[:2])
        parts.append(f"Has worked at {company_str}.")

    all_skills = []
    for skill_list in skills_by_category.values():
        all_skills.extend(skill_list[:2])
    if all_skills:
        parts.append(f"Skilled in {', '.join(all_skills[:6])}.")

    edu_text = sections.get("education", "").strip()
    if edu_text:
        edu_first = edu_text.split("\n")[0].strip()
        if edu_first:
            parts.append(f"Educated at {edu_first[:60]}.")

    return limit_words(" ".join(parts), max_words=50) if parts else ""


# ──────────────────────────────────────────
# 🎯 Main Parser
# ──────────────────────────────────────────
def parse_resume(text):
    sections  = extract_sections(text)
    name      = extract_name(text)
    email     = extract_email(text)
    phone     = extract_phone(text)
    years_exp = extract_years_of_experience(sections["experience"])
    companies = extract_companies(sections["experience"])

    return {
        "name":                name,
        "email":               email,
        "phone":               phone,
        "years_of_experience": years_exp,
        "companies":           companies,
        "sections": {
            section: limit_words(content, max_words=50) if content else ""
            for section, content in sections.items()
        },
        # Raw sections passed back so app.py can call build_summary()
        "_raw_sections": sections,
    }