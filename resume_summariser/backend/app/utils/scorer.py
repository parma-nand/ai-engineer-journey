from app.models.resume import ResumeScore


def score_resume(
    basic_info: dict,
    links: dict,
    experience: list,
    skills: dict,
    education: list,
    projects: list,
    certifications: list,
    sections: dict,
) -> ResumeScore:
    score = 0
    missing = []
    suggestions = []

    # ── Basic Info (20 pts) ───────────────────────────────────────
    if basic_info.get("name"):
        score += 5
    else:
        missing.append("Name")
    if basic_info.get("email"):
        score += 5
    else:
        missing.append("Email")
    if basic_info.get("phone"):
        score += 5
    else:
        missing.append("Phone number")
    if basic_info.get("location"):
        score += 5
    else:
        suggestions.append("Add your location/city to improve ATS ranking")

    # ── Links (10 pts) ────────────────────────────────────────────
    has_links = bool(links.get("linkedin") or links.get("github"))
    if links.get("linkedin"):
        score += 5
    else:
        suggestions.append("Add LinkedIn profile URL")
    if links.get("github"):
        score += 5
    else:
        suggestions.append("Add GitHub profile URL")

    # ── Summary (10 pts) ──────────────────────────────────────────
    has_summary = bool(sections.get("summary", "").strip())
    if has_summary:
        score += 10
    else:
        missing.append("Professional Summary")
        suggestions.append("Add a 3–5 line professional summary at the top")

    # ── Experience (25 pts) ───────────────────────────────────────
    exp_clarity = 0
    if experience:
        score += 10
        exp_clarity += 50
        # Check bullet points exist
        has_bullets = any(len(b.get("description", [])) > 0 for b in experience)
        if has_bullets:
            score += 10
            exp_clarity += 30
        else:
            suggestions.append("Add bullet-point achievements under each role")
        # Quantified achievements
        all_desc = " ".join(
            d for b in experience for d in b.get("description", [])
        )
        if re.search(r'\d+%|\d+x|\$\d+|\d+ (years|months|users|clients)', all_desc):
            score += 5
            exp_clarity += 20
        else:
            suggestions.append("Quantify achievements (e.g., 'improved speed by 30%')")
    else:
        missing.append("Work Experience")

    # ── Skills (15 pts) ───────────────────────────────────────────
    skills_depth = 0
    total_skills = sum(len(v) for v in skills.values())
    if total_skills >= 10:
        score += 15
        skills_depth = 100
    elif total_skills >= 5:
        score += 10
        skills_depth = 60
        suggestions.append("Add more skills to improve discoverability")
    elif total_skills > 0:
        score += 5
        skills_depth = 30
        suggestions.append("Expand your skills section with relevant technologies")
    else:
        missing.append("Skills")

    # ── Education (10 pts) ────────────────────────────────────────
    if education:
        score += 10
    else:
        missing.append("Education")

    # ── Projects (5 pts) ──────────────────────────────────────────
    has_projects = bool(projects)
    if projects:
        score += 5
    else:
        suggestions.append("Add personal/open-source projects to stand out")

    # ── Certifications (5 pts) ────────────────────────────────────
    has_certifications = bool(certifications)
    if certifications:
        score += 5
    else:
        suggestions.append("Add relevant certifications (AWS, Google, etc.)")

    return ResumeScore(
        overall=min(score, 100),
        has_summary=has_summary,
        has_links=has_links,
        has_projects=has_projects,
        has_certifications=has_certifications,
        experience_clarity=min(exp_clarity, 100),
        skills_depth=skills_depth,
        missing_sections=missing,
        suggestions=suggestions[:5],  # top 5 suggestions
    )


# Need re for the scoring function
import re