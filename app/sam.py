import streamlit as st
import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_internships(candidate_profile_file, dataset_file="synthetic_internships.csv", top_k=10):
    """
    Recommend internships based on candidate profile using equal-weighted score for skills, degree, and location.
    Returns: top_k_df, precision, recall, f1_score
    """
    # Load candidate
    with open(candidate_profile_file, "r") as f:
        candidate = json.load(f)

    # Load internship dataset
    df = pd.read_csv(dataset_file)

    # Preprocess
    df["skills_required_processed"] = df["skills_required"].str.lower()
    df["degrees_eligible_processed"] = df["degrees_eligible"].str.lower()
    df["location_processed"] = df["location_mode"].str.lower()
    candidate_skills_str = " ".join(candidate["skills"])

    # TF-IDF for skills similarity
    tfidf = TfidfVectorizer()
    all_skills = df["skills_required_processed"].tolist() + [candi
# Advanced Modelsdate_skills_str]
    tfidf_matrix = tfidf.fit_transform(all_skills)
    cos_sim = np.array(tfidf_matrix[-1].dot(tfidf_matrix[:-1].T).todense()).flatten()
    df["skill_score"] = cos_sim

    # Normalize skill_score to 0–1
    df["skill_score_norm"] = (df["skill_score"] - df["skill_score"].min()) / (df["skill_score"].max() - df["skill_score"].min() + 1e-6)

    # Degree & Location score (0 or 1)
    df["degree_score"] = df["degrees_eligible_processed"].apply(
        lambda x: 1 if candidate["degree"].lower() in x else 0
    )
    df["location_score"] = df["location_processed"].apply(
        lambda x: 1 if candidate["preferred_location"].lower() in x or "remote" in x else 0
    )

    # Final equal-weighted score
    df["final_score"] = (df["skill_score_norm"] + df["degree_score"] + df["location_score"]) / 3

    # -----------------------------
    # Simulated ground truth for metrics
    # -----------------------------
# Advanced Models
    def is_relevant(row):
        candidate_skills_set = set([s.lower() for s in candidate["skills"]])
        internship_skills_set = set([s.strip() for s in row["skills_required"].lower().split(",")])
        skill_overlap = len(candidate_skills_set.intersection(internship_skills_set))
        return 1 if skill_overlap >= 2 and candidate["degree"].lower() in row["degrees_eligible"].lower() else 0

    df["ground_truth"] = df.apply(is_relevant, axis=1)

    # Metrics for top-K
    top_k_df = df.sort_values("final_score", ascending=False).head(top_k)
    y_true = top_k_df["ground_truth"]
    y_pred = [1] * top_k

    precision_at_k = precision_score(y_true, y_pred, zero_division=0)
    recall_at_k = recall_score(y_true, y_pred, zero_division=0)
    f1_at_k = f1_score(y_true, y_pred, zero_division=0)

    # Return top-K internships and metrics
    return top_k_df, precision_at_k, recall_at_k, f1_at_k


# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(
    page_title="PM Internship Recommendation Engine",
    layout="centered",
    page_icon="🎯"
)

# Header
st.markdown("""
    <div style="background-color:#7b5fcf;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">PM Internship Recommendation Engine</h2>
        <p style="color:white;text-align:center;">Student Profile Registration</p>
    </div>
""", unsafe_allow_html=True)

# Candidate Form
with st.form("candidate_form", clear_on_submit=False):
    st.subheader("👤 Basic Information")
    name = st.text_input("Full Name", key="name_input")

    st.subheader("🎓 Academic Information")
    degree = st.selectbox("Degree", ["Select Degree", "BE", "B.Tech", "BSc", "MSc", "MBA", "Other"])
    year = st.selectbox("Year of Study", ["Select Year", "1", "2", "3", "4"])
    domain = st.selectbox("Domain / Specialization", ["Select Domain", "Computer Science & Engineering", "Information Technology", "Mechanical Engineering", "Civil Engineering", "Electrical & Electronics Engineering", "Electronics & Communication Engineering", "Artificial Intelligence & Machine Learning", "Cybersecurity", "Data Science", "Biotechnology", "Robotics & Automation", "Other"])

    st.subheader("💡 Skills")
    skills = st.text_area("Enter your skills (comma-separated, e.g., Python, SQL, CAD)")

    st.subheader("📍 Internship Preferences")
    location = st.selectbox("Preferred Location", ["Select Location", "Remote", "Tamil Nadu", "Karnataka", "Kerala", "Other"])

    st.subheader("📞 Contact Details")
    email = st.text_input("Email ID", key="email_input")
    phone = st.text_input("Phone Number", key="phone_input")

    submitted = st.form_submit_button("Submit")

# Handle Form Submission
if submitted:
    if not name or degree=="Select Degree" or year=="Select Year" or domain=="Select Domain" or not skills or location=="Select Location" or not email or not phone:
        st.error("⚠️ Please fill in all fields before submitting!")
    else:
        skills_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
        profile = {
            "name": name,
            "degree": degree,
            "year_of_study": year,
            "domain": domain,
            "skills": skills_list,
            "preferred_location": location,
            "email": email,
            "phone": phone
        }

        os.makedirs("data", exist_ok=True)
        filename = f"data/profile_{name.replace(' ', '_').lower()}.json"
        with open(filename, "w") as f:
            json.dump(profile, f, indent=4)

        st.success("✅ Candidate Profile Saved Successfully!")
        st.json(profile)
        st.info(f"💾 Saved to: {filename}")

        # Recommendations
        if not os.path.exists("synthetic_internships.csv"):
            st.warning("⚠️ Internship dataset not found. Please ensure 'synthetic_internships.csv' exists.")
        else:
            top_k_df, precision, recall, f1 = recommend_internships(filename, "synthetic_internships.csv", top_k=10)
            st.subheader("🎯 Top Internship Recommendations")
            st.dataframe(top_k_df[["company_name", "intern_role", "stipend", "location_mode", "skills_required", "degrees_eligible", "final_score"]], use_container_width=True)

            st.subheader("📊 Model Metrics (Top-10 Recommendations)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision@10", f"{precision:.2f}")
            col2.metric("Recall@10", f"{recall:.2f}")
            col3.metric("F1-Score@10", f"{f1:.2f}")
