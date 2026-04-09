import re
import spacy
from datetime import datetime
from dateutil import parser as date_parser

nlp = spacy.load("en_core_web_sm")


# 📧 Email
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
    return match.group(0) if match else None


# 📞 Phone
def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-]{8,15}\d", text)
    return match.group(0) if match else None


# 👤 Name (NER)
def extract_name(text):
    doc = nlp(text[:1000])  # first part is enough
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


# 📂 Section Split
def extract_sections(text):
    sections = {
        "experience": "",
        "education": "",
        "projects": "",
        "certifications": ""
    }

    current = None
    for line in text.split("\n"):
        line_lower = line.lower()

        if "experience" in line_lower:
            current = "experience"
        elif "education" in line_lower:
            current = "education"
        elif "project" in line_lower:
            current = "projects"
        elif "certification" in line_lower:
            current = "certifications"

        elif current:
            sections[current] += line + "\n"

    return sections


# 📅 Experience Calculation
def extract_years_of_experience(exp_text):
    dates = re.findall(r"([A-Za-z]{3,9}\s?\d{4})", exp_text)

    parsed_dates = []
    for d in dates:
        try:
            parsed_dates.append(date_parser.parse(d))
        except:
            pass

    if len(parsed_dates) >= 2:
        start = min(parsed_dates)
        end = max(parsed_dates)
        return round((end - start).days / 365, 1)

    return 0


# 🎯 Main Parser
def parse_resume(text):
    sections = extract_sections(text)

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "sections": sections,
        "years_of_experience": extract_years_of_experience(sections["experience"])
    }