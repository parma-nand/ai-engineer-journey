import base64
import streamlit as st
import requests
import os

# 🔗 API
API_URL = os.getenv("API_URL", "http://resume-api:8000/parse")

st.set_page_config(page_title="Resume Parser", layout="centered")

# 🎯 Title
st.title("📄 Resume Parser")

# 📤 Upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# 🔘 Button
if st.button("Parse Resume"):

    if uploaded_file is not None:

        with st.spinner("Processing..."):

            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }
                )

                if response.status_code == 200:

                    result = response.json()
                    data = result.get("data", {})

                    # =========================
                    # 👤 Candidate Info (Simple & Clean)
                    # =========================
                    st.markdown("## Candidate Information")

                    st.markdown(f"**Name is name:** {data.get('name', 'N/A')}")
                    st.markdown(f"**Email:** {data.get('email', 'N/A')}")
                    st.markdown(f"**Phone:** {data.get('phone', 'N/A')}")
                    st.markdown(f"**Experience:** {data.get('years_of_experience', 0)} years")

                    st.divider()
                    
                    # =========================
                    # 🎯 Skills (Category-wise)
                    # =========================
                    st.markdown("## Skills")
                    skills = result.get("skills_by_category", {})
                    if skills:
                        for category, skill_list in skills.items():
                            if skill_list:
                                st.markdown(f"**{category}:** {', '.join(skill_list)}")
                    else :
                        st.write("No Skills Found")

                    # =========================
                    # 📂 Sections (Static, limited text)
                    # =========================
                    
                    st.divider()
                    st.markdown("## Resume Sections")

                    def limit_words(text, max_words=50):
                        words = text.split()
                        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

                    sections = data.get("sections", {})

                    for section, content in sections.items():
                        st.markdown(f"### {section.title()}")
                        if content:
                            st.write(limit_words(content))
                        else:
                            st.write("Not found")
                        st.divider()

                else:
                    st.error(f"API Error: {response.status_code}")

            except Exception as e:
                st.error(f"Connection error: {e}")

    else:
        st.warning("Please upload a PDF first")