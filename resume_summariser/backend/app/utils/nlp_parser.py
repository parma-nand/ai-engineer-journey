import re
import spacy
from dateutil import parser as date_parser
from datetime import datetime
from loguru import logger
from typing import Optional

nlp = spacy.load("en_core_web_sm")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────
SECTION_KEYWORDS = {
    "summary":        ["summary", "objective", "profile", "about me",
                       "professional summary", "executive summary", "career objective"],
    "experience":     ["experience", "work history", "employment",
                       "professional experience", "work experience"],
    "education":      ["education", "academic", "qualification", "academics"],
    "projects":       ["project", "projects", "personal projects", "key projects"],
    "certifications": ["certification", "certifications", "certificate",
                       "courses", "training", "licenses"],
    "skills":         ["skills", "technical skills", "core competencies",
                       "technologies", "tech stack"],
}

def detect_section(line: str) -> Optional[str]:
    line_lower = line.lower().strip()
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', line_lower):
                return section
    return None

def extract_sections(text: str) -> dict:
    sections = {k: "" for k in SECTION_KEYWORDS}
    current = None
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                sections[current] += "\n"
            continue
        detected = detect_section(stripped)
        if detected:
            current = detected
            continue
        if current:
            sections[current] += stripped + "\n"
    return {k: v.strip() for k, v in sections.items()}


# ─────────────────────────────────────────────────────────────────────────────
# BASIC INFO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
_HEADING_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective",
    "contact", "address", "email", "phone", "linkedin", "github", "education",
    "experience", "skills", "projects", "achievements", "certifications",
    "declaration", "professional", "technical", "languages", "interests",
}
_CONTACT_PATTERNS = re.compile(
    r"@|linkedin|github|http|www\.|portfolio|\+?\(?\d{3,}|\|",
    re.IGNORECASE
)

def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None

def extract_phone(text: str) -> Optional[str]:
    match = re.search(r"[\(\+]?\d[\d\s\-\(\)\.]{7,15}\d", text)
    if not match:
        return None
    phone = re.sub(r"[\(\)]", "", match.group(0)).strip()
    phone = re.sub(r"\s+", " ", phone).strip()
    return phone

def extract_location(text: str) -> Optional[str]:
    """Extract city/state/country from first few lines or contact section."""
    lines = [l.strip() for l in text.split("\n") if l.strip()][:10]
    # Common location patterns: "City, State" or "City, Country"
    loc_pattern = re.compile(
        r'\b([A-Z][a-zA-Z\s]+,\s*(?:[A-Z]{2}|[A-Z][a-zA-Z]+))\b'
    )
    for line in lines:
        if _CONTACT_PATTERNS.search(line):
            m = loc_pattern.search(line)
            if m:
                return m.group(1).strip()
    return None

def extract_name(text: str) -> Optional[str]:
    all_lines = [l.strip() for l in text.split("\n") if l.strip()][:6]
    candidate_lines = [l for l in all_lines if not _CONTACT_PATTERNS.search(l)]

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
        if {w.lower() for w in words}.intersection(_HEADING_WORDS):
            continue
        return line.title() if line.isupper() else line
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LINKS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_links(text: str) -> dict:
    linkedin = None
    github = None
    portfolio = []

    # LinkedIn
    li = re.search(r'linkedin\.com/in/([^\s|,\)]+)', text, re.IGNORECASE)
    if li:
        linkedin = "https://linkedin.com/in/" + li.group(1).rstrip('/')

    # GitHub
    gh = re.search(r'github\.com/([^\s|,\)]+)', text, re.IGNORECASE)
    if gh:
        path = gh.group(1).rstrip('/')
        # exclude github.com/name/repo style — keep profile only
        if path.count('/') == 0:
            github = "https://github.com/" + path
        else:
            github = "https://github.com/" + path.split('/')[0]

    # Portfolio / other URLs
    urls = re.findall(
        r'https?://(?!linkedin|github)[^\s|,\)>]+', text, re.IGNORECASE
    )
    portfolio = list(set(urls))[:3]

    return {"linkedin": linkedin, "github": github, "portfolio": portfolio}


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
_DATE_RANGE_PAT = re.compile(
    r"(\d{1,2}/\d{4}|[A-Za-z]{3,9}\.?\s?\d{4}|\d{4})"
    r"\s*[-–—to]+\s*"
    r"(\d{1,2}/\d{4}|[A-Za-z]{3,9}\.?\s?\d{4}|\d{4}|present|current|now)",
    re.IGNORECASE
)

_ROLE_WORDS = re.compile(
    r"^(developer|engineer|analyst|manager|consultant|architect|intern|lead|"
    r"senior|junior|associate|specialist|designer|director|officer|executive|"
    r"head|president|vp|cto|ceo|sde|swe|devops|qa|tester|scrum)",
    re.IGNORECASE
)

_COMPANY_SUFFIXES = re.compile(
    r"\b(pvt\.?\s*ltd\.?|ltd\.?|llc|inc\.?|corp\.?|technologies|tech|solutions|"
    r"services|systems|consulting|software|global|group|foundation|ventures|"
    r"associates|enterprises|limited|pvt|private)\b",
    re.IGNORECASE
)

_TECH_TERMS = {
    "java", "spring", "spring boot", "react", "react.js", "node", "nodejs",
    "html", "css", "sql", "aws", "gcp", "azure", "docker", "kubernetes",
    "python", "django", "flask", "fastapi", "hibernate", "jpa", "jdbc",
    "bootstrap", "javascript", "typescript", "postman", "git", "github",
    "mysql", "postgresql", "mongodb", "redis", "kafka", "jenkins", "maven",
    "swagger", "mockito", "selenium", "agile", "rest", "restful", "api",
    "linux", "unix", "jira", "terraform", "microservices", "ci/cd",
}

def _parse_date(date_str: str) -> Optional[datetime]:
    try:
        normalized = re.sub(r"[-/]", " ", date_str)
        return date_parser.parse(normalized, default=date_parser.parse("Jan 2000"))
    except Exception:
        return None

def _months_between(start: datetime, end: datetime) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))

def extract_experience_blocks(exp_text: str) -> list[dict]:
    """Parse experience section into structured job blocks."""
    if not exp_text:
        return []

    lines = [l.strip() for l in exp_text.split("\n") if l.strip()]
    blocks = []
    current_block = None

    for i, line in enumerate(lines):
        date_match = _DATE_RANGE_PAT.search(line)

        if date_match:
            # Save previous block
            if current_block:
                blocks.append(current_block)

            start_str = date_match.group(1)
            end_str   = date_match.group(2)
            is_current = bool(re.match(r"present|current|now", end_str, re.IGNORECASE))

            start_dt = _parse_date(start_str)
            end_dt   = datetime.today() if is_current else _parse_date(end_str)
            duration = _months_between(start_dt, end_dt) if start_dt and end_dt else None

            # Look back for company and role in previous 1–3 lines
            company, role = _extract_company_role(lines, i)

            current_block = {
                "company":        company or "Unknown",
                "role":           role,
                "start_date":     start_str,
                "end_date":       "Present" if is_current else end_str,
                "is_current":     is_current,
                "duration_months": duration,
                "description":    [],
            }
        elif current_block and line.startswith(("•", "-", "*", "–", "+")):
            # Bullet point → job description
            clean = line.lstrip("•-*–+ ").strip()
            if clean:
                current_block["description"].append(clean)
        elif current_block and not _DATE_RANGE_PAT.search(line):
            # Could be continuation of description (long line)
            if len(line.split()) > 6:
                current_block["description"].append(line)

    if current_block:
        blocks.append(current_block)

    return blocks

def _extract_company_role(lines: list, date_line_idx: int):
    """Look above a date-range line to find company name and role."""
    company = None
    role = None
    candidates = lines[max(0, date_line_idx - 3):date_line_idx]

    # The line immediately before the date is often the role or company+role combined
    for candidate in reversed(candidates):
        if not candidate or candidate.startswith(("•", "-", "*")):
            continue
        if len(candidate.split()) > 10:
            continue

        # Strip parenthetical role info
        clean = re.sub(r"\(.*?\)", "", candidate).strip(" ,-|")
        if not clean:
            continue

        # If line has company suffix → it's the company line
        if _COMPANY_SUFFIXES.search(clean):
            company = clean
            continue

        # If starts with a role word → it's the role line
        if _ROLE_WORDS.match(clean):
            role = clean
            continue

        # spaCy ORG NER check
        doc = nlp(clean)
        for ent in doc.ents:
            if ent.label_ == "ORG" and _is_valid_org(ent.text):
                company = ent.text.strip()

        # If still no company and line looks like a proper name (Title Case, short)
        if not company and clean[0].isupper() and len(clean.split()) <= 5:
            if not _ROLE_WORDS.match(clean):
                company = clean

    return company, role

def _is_valid_org(name: str) -> bool:
    if not name or len(name.split()) > 6:
        return False
    if name[0].islower():
        return False
    if name.lower().strip() in _TECH_TERMS:
        return False
    return True

def extract_total_experience(blocks: list[dict]) -> float:
    """Merge overlapping intervals and sum total months → years."""
    intervals = []
    for b in blocks:
        start = _parse_date(b["start_date"]) if b.get("start_date") else None
        if b.get("is_current"):
            end = datetime.today()
        else:
            end = _parse_date(b["end_date"]) if b.get("end_date") else None
        if start and end and end > start:
            intervals.append((start, end))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 365, 1)

def get_current_company(blocks: list[dict]) -> Optional[str]:
    """Return the company with is_current=True, or the most recent one."""
    current = [b for b in blocks if b.get("is_current")]
    if current:
        return current[0]["company"]
    if blocks:
        return blocks[0]["company"]  # first block = most recent
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SKILLS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
SKILLS_CATALOG = {
    "Frontend":    ["react", "react.js", "vue", "angular", "html", "html5", "css",
                    "css3", "javascript", "typescript", "bootstrap", "tailwind",
                    "nextjs", "next.js", "redux", "jquery", "sass", "webpack"],
    "Backend":     ["java", "python", "node", "nodejs", "spring", "spring boot",
                    "fastapi", "flask", "django", "express", "hibernate", "jpa",
                    "jdbc", "maven", "gradle", "go", "rust", "php", "ruby"],
    "Database":    ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                    "oracle", "sqlite", "cassandra", "dynamodb", "h2"],
    "Cloud & DevOps": ["aws", "gcp", "azure", "docker", "kubernetes", "jenkins",
                       "ci/cd", "terraform", "ansible", "github actions", "gitlab",
                       "nginx", "linux"],
    "Tools":       ["git", "github", "jira", "postman", "swagger", "mockito",
                    "selenium", "kafka", "rabbitmq", "graphql", "rest", "restful"],
    "Data & ML":   ["machine learning", "deep learning", "tensorflow", "pytorch",
                    "keras", "pandas", "numpy", "scikit-learn", "spark", "hadoop"],
}

def extract_skills(text: str, skills_section: str = "") -> dict:
    combined = (skills_section + " " + text).lower()
    found = {}
    for category, skills in SKILLS_CATALOG.items():
        matched = []
        for skill in skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, combined):
                matched.append(skill)
        if matched:
            found[category] = matched
    return found


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
_DEGREE_KEYWORDS = re.compile(
    r'\b(b\.?tech|m\.?tech|b\.?e|m\.?e|bca|mca|b\.?sc|m\.?sc|b\.?com|mba|ph\.?d|'
    r'bachelor|master|doctorate|diploma|associate|b\.s|m\.s)\b',
    re.IGNORECASE
)
_GPA_PAT = re.compile(r'(?:gpa|cgpa|grade)[:\s]*(\d+\.?\d*)', re.IGNORECASE)
_YEAR_PAT = re.compile(r'\b(19|20)\d{2}\b')

def extract_education(edu_text: str) -> list[dict]:
    if not edu_text:
        return []

    results = []
    lines = [l.strip() for l in edu_text.split("\n") if l.strip()]
    current_edu = None

    for line in lines:
        has_degree  = _DEGREE_KEYWORDS.search(line)
        has_year    = _YEAR_PAT.search(line)
        gpa_match   = _GPA_PAT.search(line)

        if has_degree or (has_year and len(line.split()) <= 10):
            if current_edu:
                results.append(current_edu)

            # Try to extract institution via NER
            doc = nlp(line)
            institution = None
            for ent in doc.ents:
                if ent.label_ in ("ORG", "FAC"):
                    institution = ent.text.strip()
                    break
            if not institution:
                institution = line.split(",")[0].strip()

            years = _YEAR_PAT.findall(line)
            current_edu = {
                "institution": institution,
                "degree":      has_degree.group(0) if has_degree else None,
                "field":       None,
                "start_year":  years[0] if len(years) > 0 else None,
                "end_year":    years[1] if len(years) > 1 else None,
                "gpa":         gpa_match.group(1) if gpa_match else None,
            }
        elif current_edu and gpa_match:
            current_edu["gpa"] = gpa_match.group(1)

    if current_edu:
        results.append(current_edu)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
_GITHUB_URL = re.compile(r'github\.com/\S+', re.IGNORECASE)

def extract_projects(proj_text: str) -> list[dict]:
    if not proj_text:
        return []

    projects = []
    lines = [l.strip() for l in proj_text.split("\n") if l.strip()]
    current_proj = None

    for line in lines:
        is_bullet = line.startswith(("•", "-", "*", "–", "+"))
        gh_match  = _GITHUB_URL.search(line)

        if not is_bullet and len(line.split()) <= 10 and not gh_match:
            if current_proj:
                projects.append(current_proj)
            # Detect techs inline e.g. "ProjectName | Java, React"
            parts = re.split(r'\|', line, maxsplit=1)
            proj_name = parts[0].strip()
            techs = []
            if len(parts) > 1:
                techs = [t.strip() for t in parts[1].split(",")]
            current_proj = {
                "name": proj_name,
                "description": None,
                "technologies": techs,
                "link": None,
            }
        elif current_proj:
            if gh_match:
                current_proj["link"] = "https://" + gh_match.group(0)
            elif is_bullet:
                clean = line.lstrip("•-*–+ ").strip()
                if not current_proj["description"]:
                    current_proj["description"] = clean
                # Extract techs from bullet text
                for cat_skills in SKILLS_CATALOG.values():
                    for skill in cat_skills:
                        if re.search(r'\b' + re.escape(skill) + r'\b', clean, re.IGNORECASE):
                            if skill not in current_proj["technologies"]:
                                current_proj["technologies"].append(skill)

    if current_proj:
        projects.append(current_proj)

    return projects


# ─────────────────────────────────────────────────────────────────────────────
# CERTIFICATIONS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_certifications(cert_text: str) -> list[str]:
    if not cert_text:
        return []
    certs = []
    for line in cert_text.split("\n"):
        line = line.strip().lstrip("•-*–+ ")
        if line and len(line) > 3:
            certs.append(line)
    return certs[:10]


# ─────────────────────────────────────────────────────────────────────────────
# SMART SUMMARY GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def _limit_words(text: str, max_words: int = 60) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."

def generate_summary(
    name: Optional[str],
    years_exp: float,
    current_company: Optional[str],
    skills: dict,
    sections: dict,
    education: list,
) -> str:
    # Priority 1: Use candidate's own summary section
    own = sections.get("summary", "").strip()
    if own:
        return _limit_words(own, 60)

    # Priority 2: Auto-generate
    parts = []

    # Role from first line of experience
    exp_text = sections.get("experience", "")
    role = None
    if exp_text:
        for line in exp_text.split("\n"):
            line = line.strip()
            if line and not line.startswith(("•", "-")):
                candidate = _DATE_RANGE_PAT.sub("", line).strip(" ,|–")
                if candidate and len(candidate.split()) <= 6:
                    role = candidate
                    break

    intro = f"{name} is a" if name else "A"
    if role:
        parts.append(f"{intro} {role}")
    else:
        parts.append(f"{intro} software professional")

    if years_exp > 0:
        parts[-1] += f" with {years_exp} years of experience."
    else:
        parts[-1] += "."

    if current_company:
        parts.append(f"Currently at {current_company}.")

    all_skills = []
    for skill_list in skills.values():
        all_skills.extend(skill_list[:2])
    if all_skills:
        parts.append(f"Skilled in {', '.join(all_skills[:6])}.")

    if education:
        edu = education[0]
        if edu.get("institution"):
            parts.append(f"Holds a {edu.get('degree', 'degree')} from {edu['institution']}.")

    return _limit_words(" ".join(parts), 60)