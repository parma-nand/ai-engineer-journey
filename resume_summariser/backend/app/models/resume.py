from pydantic import BaseModel, EmailStr
from typing import Optional


class BasicInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    confidence: dict[str, float] = {}


class Links(BaseModel):
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: list[str] = []


class WorkExperience(BaseModel):
    company: str
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    duration_months: Optional[int] = None
    description: list[str] = []


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    gpa: Optional[str] = None


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: list[str] = []
    link: Optional[str] = None


class ResumeScore(BaseModel):
    overall: int                   # 0–100
    has_summary: bool
    has_links: bool
    has_projects: bool
    has_certifications: bool
    experience_clarity: int        # 0–100
    skills_depth: int              # 0–100
    missing_sections: list[str] = []
    suggestions: list[str] = []


class ParsedResume(BaseModel):
    id: str
    basic_info: BasicInfo
    links: Links
    current_company: Optional[str] = None
    experience: list[WorkExperience] = []
    total_experience_years: float = 0.0
    skills: dict[str, list[str]] = {}
    education: list[Education] = []
    projects: list[Project] = []
    certifications: list[str] = []
    summary: str = ""
    score: ResumeScore
    raw_text_preview: str = ""