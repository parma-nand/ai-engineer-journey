import streamlit as st
import requests
import os

# 🔗 Backend API URL
API_URL = os.getenv("API_URL", "http://resume-api:8000/parse")

st.set_page_config(page_title="Resume Parser", layout="centered")

st.title("📄 Resume Parser")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Parse Resume"):

    if uploaded_file is not None:

        with st.spinner("Processing your resume..."):

            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()
                    data   = result.get("data", {})

                    # ──────────────────────────────
                    # 👤 Candidate Info
                    # ──────────────────────────────
                    st.markdown("## 👤 Candidate Information")

                    st.markdown(f"**Name:** {data.get('name') or 'N/A'}")
                    st.markdown(f"**Email:** {data.get('email') or 'N/A'}")
                    st.markdown(f"**Phone:** {data.get('phone') or 'N/A'}")
                    st.markdown(f"**Experience:** {data.get('years_of_experience', 0)} years")

                    companies = data.get("companies", [])
                    if companies:
                        st.markdown(f"**{'Company' if len(companies) == 1 else 'Companies'}:** {', '.join(companies)}")

                    st.divider()

                    # ──────────────────────────────
                    # 📝 Summary
                    # ──────────────────────────────
                    summary = result.get("summary", "")
                    if summary:
                        st.markdown("## 📝 Summary")
                        st.write(summary)
                        st.divider()

                    # ──────────────────────────────
                    # 🎯 Skills by Category
                    # ──────────────────────────────
                    st.markdown("## 🎯 Skills")
                    skills = result.get("skills_by_category", {})
                    if skills:
                        for category, skill_list in skills.items():
                            if skill_list:
                                st.markdown(f"**{category}:** {', '.join(skill_list)}")
                    else:
                        st.info("No recognised skills found.")

                    st.divider()

                    # ──────────────────────────────
                    # 📂 Resume Sections
                    #    Shows all 4: experience, education,
                    #    projects, certifications
                    # ──────────────────────────────
                    st.markdown("## 📂 Resume Sections")

                    sections = data.get("sections", {})

                    SECTION_ICONS = {
                        "experience":     "💼",
                        "education":      "🎓",
                        "projects":       "🛠️",
                        "certifications": "📜",
                    }

                    for section_key in ["experience", "education", "projects", "certifications"]:
                        icon    = SECTION_ICONS.get(section_key, "•")
                        title   = section_key.title()
                        content = sections.get(section_key, "").strip()

                        st.markdown(f"### {icon} {title}")
                        if content:
                            st.write(content)
                        else:
                            st.caption("Not found in resume.")
                        st.divider()

                elif response.status_code == 422:
                    detail = response.json().get("detail", "Unprocessable file.")
                    st.error(f"❌ {detail}")

                elif response.status_code == 400:
                    detail = response.json().get("detail", "Bad request.")
                    st.error(f"❌ {detail}")

                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the backend. Is it running?")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Try again.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    else:
        st.warning("⚠️ Please upload a PDF first.")