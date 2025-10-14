import pandas as pd
import random
import os

# -----------------------------
# Synthetic Data Options
# -----------------------------
company_names = [
    "Infosys", "TCS", "Wipro", "HCL", "Accenture", "Google", "Microsoft",
    "Amazon", "IBM", "Flipkart", "Byjus", "Cognizant", "Capgemini", "Tech Mahindra"
]

intern_roles = [
    "Data Science Intern", "Machine Learning Intern", "Web Development Intern",
    "Mobile App Development Intern", "UI/UX Design Intern", "Cybersecurity Intern",
    "Cloud Engineering Intern", "Business Analyst Intern", "Digital Marketing Intern",
    "Robotics Intern", "Embedded Systems Intern"
]

locations = [
    "Remote", "Tamil Nadu", "Karnataka", "Kerala", "Andhra Pradesh", "Telangana",
    "Maharashtra", "Delhi", "Uttar Pradesh", "West Bengal", "Gujarat",
    "Madhya Pradesh", "Rajasthan", "Punjab", "Odisha", "Bihar", "Assam", "Chandigarh"
]

skills_pool = [
    "Python", "Java", "C++", "SQL", "Excel", "Power BI", "Tableau", "TensorFlow",
    "Keras", "PyTorch", "ReactJS", "NodeJS", "HTML/CSS", "JavaScript", "MATLAB",
    "SolidWorks", "CAD", "Linux", "AWS", "Azure", "Docker", "Kubernetes"
]

degrees_pool = [
    "BE", "B.Tech", "BSc", "MSc", "MBA"
]

# -----------------------------
# Function to generate synthetic data
# -----------------------------
def generate_synthetic_data(num_entries=500):
    data = []
    for _ in range(num_entries):
        company = random.choice(company_names)
        role = random.choice(intern_roles)
        stipend = random.choice([5000, 10000, 15000, 20000, 25000, 30000])
        location = random.choice(locations)
        skills_required = random.sample(skills_pool, k=random.randint(2, 5))
        degrees_eligible = random.sample(degrees_pool, k=random.randint(1, 3))

        entry = {
            "company_name": company,
            "intern_role": role,
            "stipend": stipend,
            "location_mode": location,
            "skills_required": ", ".join(skills_required),
            "degrees_eligible": ", ".join(degrees_eligible)
        }

        data.append(entry)
    df = pd.DataFrame(data)
    return df

# -----------------------------
# Generate & Save Dataset
# -----------------------------
df = generate_synthetic_data(num_entries=500)  # <-- Updated to 500 entries

os.makedirs("data", exist_ok=True)
file_path = "data/synthetic_internships.csv"
df.to_csv(file_path, index=False)

print(f"✅ Synthetic internship dataset generated and saved to {file_path}")
