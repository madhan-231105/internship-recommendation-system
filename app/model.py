def recommend_internships(candidate_profile_file, dataset_file="data/synthetic_internships.csv", top_k=10):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import precision_score, recall_score, f1_score
    import numpy as np
    import json

    # Load candidate
    with open(candidate_profile_file, "r") as f:
        candidate = json.load(f)

    df = pd.read_csv(dataset_file)

    # Preprocess
    df["skills_required_processed"] = df["skills_required"].str.lower()
    df["degrees_eligible_processed"] = df["degrees_eligible"].str.lower()
    df["location_processed"] = df["location_mode"].str.lower()
    candidate_skills_str = " ".join(candidate["skills"])

    # TF-IDF for skills similarity
    tfidf = TfidfVectorizer()
    all_skills = df["skills_required_processed"].tolist() + [candidate_skills_str]
    tfidf_matrix = tfidf.fit_transform(all_skills)
    cos_sim = np.array(tfidf_matrix[-1].dot(tfidf_matrix[:-1].T).todense()).flatten()
    df["skill_score"] = cos_sim

    # Degree & Location score
    df["degree_score"] = df["degrees_eligible_processed"].apply(
        lambda x: 1 if candidate["degree"].lower() in x else 0
    )
    df["location_score"] = df["location_processed"].apply(
        lambda x: 1 if candidate["preferred_location"].lower() in x or "remote" in x else 0
    )

    # Final weighted score
    df["final_score"] = 0.6*df["skill_score"] + 0.2*df["degree_score"] + 0.2*df["location_score"]

    # -----------------------------
    # Simulated ground truth for metrics
    # -----------------------------
    def is_relevant(row):
        candidate_skills_set = set([s.lower() for s in candidate["skills"]])
        internship_skills_set = set([s.strip() for s in row["skills_required"].lower().split(",")])
        skill_overlap = len(candidate_skills_set.intersection(internship_skills_set))
        return 1 if skill_overlap>=2 and candidate["degree"].lower() in row["degrees_eligible"].lower() else 0

    df["ground_truth"] = df.apply(is_relevant, axis=1)

    # Metrics for top-K
    top_k_df = df.sort_values("final_score", ascending=False).head(top_k)
    y_true = top_k_df["ground_truth"]
    y_pred = [1]*top_k

    precision_at_k = precision_score(y_true, y_pred)
    recall_at_k = recall_score(y_true, y_pred)
    f1_at_k = f1_score(y_true, y_pred)

    # Return top-K internships and metrics
    return top_k_df, precision_at_k, recall_at_k, f1_at_k
