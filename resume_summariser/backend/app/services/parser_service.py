import uuid
from loguru import logger

from app.utils.pdf_extractor import extract_text_from_pdf
from app.utils.nlp_parser import (
    extract_sections,
    extract_name, extract_email, extract_phone, extract_location,
    extract_links,
    extract_experience_blocks, extract_total_experience, get_current_company,
    extract_skills,
    extract_education,
    extract_projects,
    extract_certifications,
    generate_summary,
)
from app.utils.scorer import score_resume
from app.models.resume import (
    ParsedResume, BasicInfo, Links, WorkExperience, Education, Project
)
from app.config import get_settings

settings = get_settings()


class ResumeParserService:

    def parse(self, file_path: str) -> ParsedResume:
        """Full pipeline: PDF → text → NLP → structured output."""

        # ── 1. Extract raw text ───────────────────────────────────
        logger.info(f"Extracting text from: {file_path}")
        text, error_type, page_count = extract_text_from_pdf(file_path)

        if error_type:
            raise ValueError(
                f"Cannot parse PDF: {error_type}. "
                "Please upload a text-based PDF (not scanned)."
            )

        if page_count > settings.MAX_PAGES:
            raise ValueError(
                f"Resume has {page_count} pages. Maximum allowed: {settings.MAX_PAGES}."
            )

        logger.info(f"Extracted {len(text)} chars from {page_count} pages")

        # ── 2. Section detection ──────────────────────────────────
        sections = extract_sections(text)

        # ── 3. Basic info ─────────────────────────────────────────
        name     = extract_name(text)
        email    = extract_email(text)
        phone    = extract_phone(text)
        location = extract_location(text)

        confidence = {
            "name":  0.9 if name else 0.0,
            "email": 1.0 if email else 0.0,
            "phone": 0.9 if phone else 0.0,
        }

        basic_info = BasicInfo(
            name=name, email=email, phone=phone,
            location=location, confidence=confidence
        )

        # ── 4. Links ──────────────────────────────────────────────
        raw_links = extract_links(text)
        links = Links(**raw_links)

        # ── 5. Experience ─────────────────────────────────────────
        exp_blocks = extract_experience_blocks(sections["experience"])
        total_years = extract_total_experience(exp_blocks)
        current_company = get_current_company(exp_blocks)

        experience = [
            WorkExperience(
                company        = b["company"],
                role           = b.get("role"),
                start_date     = b.get("start_date"),
                end_date       = b.get("end_date"),
                is_current     = b.get("is_current", False),
                duration_months= b.get("duration_months"),
                description    = b.get("description", []),
            )
            for b in exp_blocks
        ]

        # ── 6. Skills ─────────────────────────────────────────────
        skills = extract_skills(text, sections.get("skills", ""))

        # ── 7. Education ──────────────────────────────────────────
        edu_blocks = extract_education(sections["education"])
        education = [
            Education(
                institution = e["institution"],
                degree      = e.get("degree"),
                field       = e.get("field"),
                start_year  = e.get("start_year"),
                end_year    = e.get("end_year"),
                gpa         = e.get("gpa"),
            )
            for e in edu_blocks
        ]

        # ── 8. Projects ───────────────────────────────────────────
        proj_blocks = extract_projects(sections["projects"])
        projects = [
            Project(
                name         = p["name"],
                description  = p.get("description"),
                technologies = p.get("technologies", []),
                link         = p.get("link"),
            )
            for p in proj_blocks
        ]

        # ── 9. Certifications ─────────────────────────────────────
        certifications = extract_certifications(sections["certifications"])

        # ── 10. Summary ───────────────────────────────────────────
        summary = generate_summary(
            name=name,
            years_exp=total_years,
            current_company=current_company,
            skills=skills,
            sections=sections,
            education=edu_blocks,
        )

        # ── 11. Score ─────────────────────────────────────────────
        score = score_resume(
            basic_info     = {"name": name, "email": email, "phone": phone, "location": location},
            links          = raw_links,
            experience     = exp_blocks,
            skills         = skills,
            education      = edu_blocks,
            projects       = proj_blocks,
            certifications = certifications,
            sections       = sections,
        )

        return ParsedResume(
            id                    = str(uuid.uuid4()),
            basic_info            = basic_info,
            links                 = links,
            current_company       = current_company,
            experience            = experience,
            total_experience_years= total_years,
            skills                = skills,
            education             = education,
            projects              = projects,
            certifications        = certifications,
            summary               = summary,
            score                 = score,
            raw_text_preview      = text[:500],
        )