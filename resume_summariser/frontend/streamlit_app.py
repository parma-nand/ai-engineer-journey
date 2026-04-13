import streamlit as st
import requests
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Resume Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}
.main { background: #0a0f1e; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(99,102,241,0.1) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(99, 102, 241, 0.5); }
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Profile ── */
.profile-name {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}
.profile-company {
    color: #818cf8;
    font-size: 1rem;
    font-weight: 500;
    margin-top: 0.25rem;
}
.profile-meta {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.meta-item {
    color: #94a3b8;
    font-size: 0.875rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Score Ring ── */
.score-container {
    text-align: center;
    padding: 1rem;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.score-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* ── Skills ── */
.skill-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}
.skill-category {
    color: #475569;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.75rem;
    margin-bottom: 0.3rem;
}

/* ── Timeline ── */
.timeline-item {
    border-left: 2px solid rgba(99, 102, 241, 0.3);
    padding-left: 1.5rem;
    padding-bottom: 1.5rem;
    position: relative;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -5px;
    top: 6px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #6366f1;
}
.timeline-company {
    font-weight: 600;
    color: #e2e8f0;
    font-size: 1rem;
}
.timeline-role {
    color: #818cf8;
    font-size: 0.875rem;
    margin-top: 0.1rem;
}
.timeline-date {
    color: #475569;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.2rem;
}
.timeline-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
.bullet {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 0.3rem;
    padding-left: 0.5rem;
}
.bullet::before { content: "▸ "; color: #4f46e5; }

/* ── Summary Box ── */
.summary-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.1));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ── Links ── */
.link-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #818cf8;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    text-decoration: none;
    margin-right: 0.5rem;
    transition: all 0.2s;
}

/* ── Missing sections ── */
.missing-chip {
    display: inline-block;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    margin: 0.2rem;
}
.suggestion-item {
    color: #94a3b8;
    font-size: 0.85rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.suggestion-item::before { content: "💡 "; }

/* ── Project card ── */
.project-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.project-name {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.95rem;
}
.project-desc {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 0.25rem;
}
.project-link {
    color: #6366f1;
    font-size: 0.75rem;
    text-decoration: none;
}

/* ── Upload zone ── */
.upload-hint {
    color: #475569;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 0.5rem;
}

/* Streamlit overrides */
.stFileUploader > div { border: 2px dashed rgba(99,102,241,0.4) !important; border-radius: 12px !important; background: rgba(99,102,241,0.05) !important; }
.stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 2rem !important; font-weight: 600 !important; font-family: 'Sora', sans-serif !important; transition: opacity 0.2s !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #4f46e5, #7c3aed) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card(title_icon: str, title: str, content_fn):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title_icon} {title}</div>
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)

def score_color(score: int) -> str:
    if score >= 80: return "#34d399"
    if score >= 60: return "#fbbf24"
    return "#f87171"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>📄 Resume Summarizer</h1>
        <p>AI-powered resume parsing · spaCy NLP · Production-grade extraction</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload Section ──────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF only · max 5MB · max 5 pages)",
            type=["pdf"],
            help="Text-based PDFs only. Scanned/image PDFs are not supported.",
        )
        st.markdown('<p class="upload-hint">Drag & drop or click to browse</p>', unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="card" style="margin-top:0">
            <div class="card-title">⚡ What gets extracted</div>
            <div style="color:#94a3b8; font-size:0.82rem; line-height:1.9">
                ✓ Name, email, phone, location<br>
                ✓ LinkedIn & GitHub links<br>
                ✓ Current company & all roles<br>
                ✓ Total years of experience<br>
                ✓ Skills by category<br>
                ✓ Education details<br>
                ✓ Projects & certifications<br>
                ✓ Resume quality score
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Parse button ────────────────────────────────────────────────
    if uploaded_file:
        if st.button("🚀 Parse Resume", use_container_width=False):
            with st.spinner("Analyzing resume..."):
                progress = st.progress(0)
                try:
                    progress.progress(20)
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=30,
                    )
                    progress.progress(80)

                    if response.status_code == 200:
                        data = response.json()
                        progress.progress(100)
                        st.session_state["result"] = data
                        st.success("✅ Resume parsed successfully!")
                    else:
                        detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ {detail}")
                        progress.empty()

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure the API is running on port 8000.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ── Results ─────────────────────────────────────────────────────
    if "result" in st.session_state:
        render_results(st.session_state["result"])


def render_results(data: dict):
    st.markdown("---")
    st.markdown("## 📊 Parsed Results")

    basic   = data.get("basic_info", {})
    links   = data.get("links", {})
    score   = data.get("score", {})
    skills  = data.get("skills", {})
    exp     = data.get("experience", [])
    edu     = data.get("education", [])
    projects= data.get("projects", [])
    certs   = data.get("certifications", [])
    summary = data.get("summary", "")
    current = data.get("current_company")
    total_y = data.get("total_experience_years", 0)

    # ── Row 1: Profile + Score ──────────────────────────────────────
    col_profile, col_score = st.columns([3, 1])

    with col_profile:
        name     = basic.get("name", "Unknown")
        email    = basic.get("email", "—")
        phone    = basic.get("phone", "—")
        location = basic.get("location", "—")

        st.markdown(f"""
        <div class="card">
            <div class="card-title">👤 PROFILE OVERVIEW</div>
            <div class="profile-name">{name}</div>
            <div class="profile-company">🏢 {current or 'Company not detected'}</div>
            <div class="profile-meta">
                <span class="meta-item">✉️ {email}</span>
                <span class="meta-item">📱 {phone}</span>
                <span class="meta-item">📍 {location}</span>
                <span class="meta-item">⏱️ {total_y} yrs experience</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Links
        if links.get("linkedin") or links.get("github"):
            links_html = ""
            if links.get("linkedin"):
                links_html += f'<a class="link-btn" href="{links["linkedin"]}" target="_blank">🔗 LinkedIn</a>'
            if links.get("github"):
                links_html += f'<a class="link-btn" href="{links["github"]}" target="_blank">🐙 GitHub</a>'
            for url in links.get("portfolio", []):
                links_html += f'<a class="link-btn" href="{url}" target="_blank">🌐 Portfolio</a>'
            st.markdown(f'<div style="margin-top:-0.5rem; margin-bottom:1rem">{links_html}</div>', unsafe_allow_html=True)

    with col_score:
        overall = score.get("overall", 0)
        color = score_color(overall)
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div class="card-title" style="justify-content:center">🎯 RESUME SCORE</div>
            <div class="score-number" style="color:{color}">{overall}</div>
            <div class="score-label">out of 100</div>
            <div style="margin-top:1rem; font-size:0.75rem; color:#475569">
                {'🟢 Strong' if overall >= 80 else '🟡 Average' if overall >= 60 else '🔴 Needs work'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary ─────────────────────────────────────────────────────
    if summary:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✨ PROFESSIONAL SUMMARY</div>
            <div class="summary-box">{summary}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 2: Skills + Experience ──────────────────────────────────
    col_skills, col_exp = st.columns([1, 1])

    with col_skills:
        if skills:
            skills_html = '<div class="card"><div class="card-title">🧠 SKILLS BY CATEGORY</div>'
            for category, skill_list in skills.items():
                skills_html += f'<div class="skill-category">{category}</div>'
                for skill in skill_list:
                    skills_html += f'<span class="skill-chip">{skill}</span>'
            skills_html += '</div>'
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">🧠 SKILLS</div><span style="color:#475569">No skills detected</span></div>', unsafe_allow_html=True)

    with col_exp:
        if exp:
            exp_html = '<div class="card"><div class="card-title">💼 EXPERIENCE TIMELINE</div>'
            for job in exp:
                badge = '<span class="timeline-badge">CURRENT</span>' if job.get("is_current") else ""
                role = job.get("role") or ""
                company = job.get("company", "Unknown")
                start = job.get("start_date", "")
                end = job.get("end_date", "")
                months = job.get("duration_months")
                duration_str = f" · {months//12}y {months%12}m" if months else ""

                exp_html += f"""
                <div class="timeline-item">
                    <div class="timeline-company">{company}{badge}</div>
                    <div class="timeline-role">{role}</div>
                    <div class="timeline-date">{start} → {end}{duration_str}</div>
                """
                for bullet in job.get("description", [])[:2]:
                    exp_html += f'<div class="bullet">{bullet[:100]}</div>'
                exp_html += "</div>"
            exp_html += '</div>'
            st.markdown(exp_html, unsafe_allow_html=True)

    # ── Row 3: Education + Projects ─────────────────────────────────
    col_edu, col_proj = st.columns([1, 1])

    with col_edu:
        if edu:
            edu_html = '<div class="card"><div class="card-title">🎓 EDUCATION</div>'
            for e in edu:
                gpa_str = f" · GPA {e['gpa']}" if e.get("gpa") else ""
                years = f"{e.get('start_year','?')} – {e.get('end_year','?')}"
                edu_html += f"""
                <div style="margin-bottom:1rem">
                    <div style="color:#e2e8f0; font-weight:600">{e['institution']}</div>
                    <div style="color:#818cf8; font-size:0.85rem">{e.get('degree','') or ''}{gpa_str}</div>
                    <div style="color:#475569; font-size:0.75rem; font-family:'JetBrains Mono',monospace">{years}</div>
                </div>
                """
            edu_html += '</div>'
            st.markdown(edu_html, unsafe_allow_html=True)

    # with col_proj:
    #    if projects:
    #         proj_html = '<div class="card"><div class="card-title">🚀 PROJECTS</div>'
    #         for p in projects[:4]:
    #             link_html = f'<a class="project-link" href="{p["link"]}" target="_blank">↗ View</a>' if p.get("link") else ""
    #             techs = " ".join(f'<span class="skill-chip" style="font-size:0.7rem">{t}</span>' for t in p.get("technologies", [])[:4])
    #             proj_html += f"""
    #             <div class="project-card">
    #                 <div style="display:flex;justify-content:space-between;align-items:start">
    #                     <div class="project-name">{p['name']}</div>
    #                     {link_html}
    #                 </div>
    #                 <div class="project-desc">{(p.get('description') or '')[:100]}</div>
    #                 <div style="margin-top:0.4rem">{techs}</div>
    #             </div>
    #             """
    #         proj_html += '</div>'
    #         st.markdown(proj_html, unsafe_allow_html=True)

    # ── Certifications ──────────────────────────────────────────────
    if certs:
        certs_html = '<div class="card"><div class="card-title">🏆 CERTIFICATIONS</div>'
        for c in certs:
            certs_html += f'<div style="color:#94a3b8; font-size:0.85rem; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05)">📜 {c}</div>'
        certs_html += '</div>'
        st.markdown(certs_html, unsafe_allow_html=True)

    # ── Score Breakdown + Suggestions ───────────────────────────────
    col_missing, col_suggest = st.columns([1, 1])

    with col_missing:
        missing = score.get("missing_sections", [])
        if missing:
            missing_html = '<div class="card"><div class="card-title">⚠️ MISSING SECTIONS</div>'
            for m in missing:
                missing_html += f'<span class="missing-chip">{m}</span>'
            missing_html += '</div>'
            st.markdown(missing_html, unsafe_allow_html=True)

    with col_suggest:
        suggestions = score.get("suggestions", [])
        if suggestions:
            sug_html = '<div class="card"><div class="card-title">💡 IMPROVEMENT TIPS</div>'
            for s in suggestions:
                sug_html += f'<div class="suggestion-item">{s}</div>'
            sug_html += '</div>'
            st.markdown(sug_html, unsafe_allow_html=True)

    # ── Raw JSON toggle ─────────────────────────────────────────────
    with st.expander("🔧 View Raw JSON Response"):
        st.json(data)


if __name__ == "__main__":
    main()